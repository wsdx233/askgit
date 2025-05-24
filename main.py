#!/usr/bin/env python3
# askgit.py

import argparse
import configparser
import os
import subprocess
import sys
from pathlib import Path

try:
    # OpenAI specific error can be useful if we need finer-grained error handling for OpenAI
    # from openai import OpenAIError 
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain, create_history_aware_retriever
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.messages import SystemMessage, HumanMessage
    from tqdm import tqdm
except ImportError:
    print("Core Langchain/OpenAI libraries not found. Please install them by running:")
    print("pip install langchain langchain-openai faiss-cpu tiktoken python-dotenv configparser tqdm")
    sys.exit(1)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    LANGCHAIN_GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_GOOGLE_GENAI_AVAILABLE = False

# --- Constants ---
ASKGIT_DIR_NAME = ".askgit"
CONFIG_FILE_NAME = "config.ini"
DB_DIR_NAME = "db"
DOCS_DIR_NAME = "docs"
FAISS_INDEX_NAME = "git_kb"
MAX_CONTEXT_CHARS = 350_000 
BINARY_LIKE_EXTENSIONS = [
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.tiff', '.webp',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.odt', '.ods', '.odp',
    '.zip', '.gz', '.tar', '.rar', '.7z', '.bz2', '.xz',
    '.exe', '.dll', '.so', '.o', '.a', '.dylib', '.jar', '.war', '.ear',
    '.class', '.pyc', '.pyd',
    '.mp3', '.mp4', '.mov', '.avi', '.mkv', '.flv', '.wav', '.ogg',
    '.wasm',
    '.lockb', '.sqlite', '.db', '.dat', '.bin',
    '.svg', '.eot', '.ttf', '.woff', '.woff2'
]

# --- Configuration Handling ---

def get_git_root():
    try:
        root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], universal_newlines=True).strip()
        return Path(root)
    except subprocess.CalledProcessError:
        return None

def get_askgit_dir(git_root):
    return git_root / ASKGIT_DIR_NAME

def get_config_file_path(askgit_dir):
    return askgit_dir / CONFIG_FILE_NAME

def get_db_path(askgit_dir):
    return askgit_dir / DB_DIR_NAME / FAISS_INDEX_NAME

def get_docs_dir(askgit_dir):
    docs_path = askgit_dir / DOCS_DIR_NAME
    docs_path.mkdir(parents=True, exist_ok=True)
    return docs_path

def ensure_askgit_dir_and_config(git_root):
    askgit_dir = get_askgit_dir(git_root)
    askgit_dir.mkdir(parents=True, exist_ok=True)
    (askgit_dir / DB_DIR_NAME).mkdir(parents=True, exist_ok=True)
    get_docs_dir(askgit_dir)

    config_file_path = get_config_file_path(askgit_dir)
    config = configparser.ConfigParser()

    if not config_file_path.exists():
        print("Configuration file not found. Let's set it up.")
        
        assistant_api_provider = ""
        while assistant_api_provider not in ["openai", "gemini"]:
            assistant_api_provider = input("Which API provider for ASSISTANT model? (openai/gemini): ").strip().lower()
        
        embedding_api_provider = ""
        while embedding_api_provider not in ["openai", "gemini"]:
            embedding_api_provider = input("Which API provider for EMBEDDING model? (openai/gemini): ").strip().lower()

        config['GENERAL'] = {
            'ASSISTANT_API_PROVIDER': assistant_api_provider,
            'EMBEDDING_API_PROVIDER': embedding_api_provider
        }

        openai_selected = (assistant_api_provider == "openai" or embedding_api_provider == "openai")
        gemini_selected = (assistant_api_provider == "gemini" or embedding_api_provider == "gemini")

        openai_config_data = {}
        gemini_config_data = {}

        if openai_selected:
            print("\n--- OpenAI Configuration ---")
            openai_config_data['API_KEY'] = input("Enter OpenAI API Key: ").strip()
            base_url_input = input("Enter OpenAI Base URL (e.g., https://api.openai.com/v1 or leave blank for default): ").strip()
            openai_config_data['BASE_URL'] = base_url_input if base_url_input else "https://api.openai.com/v1"
            
            if assistant_api_provider == "openai":
                openai_config_data['ASSISTANT_MODEL'] = input("Enter OpenAI Assistant Model Name (e.g., gpt-4o): ").strip()
            if embedding_api_provider == "openai":
                openai_config_data['EMBEDDING_MODEL'] = input("Enter OpenAI Embedding Model Name (e.g., text-embedding-3-small): ").strip()
            config['OPENAI'] = openai_config_data

        if gemini_selected:
            if not LANGCHAIN_GOOGLE_GENAI_AVAILABLE:
                print("\nError: 'langchain-google-genai' package is not installed, but Gemini was selected.")
                print("Please install it by running: pip install langchain-google-genai")
                sys.exit(1)
            print("\n--- Gemini Configuration ---")
            gemini_config_data['API_KEY'] = input("Enter Google API Key (for Gemini): ").strip()

            if assistant_api_provider == "gemini":
                gemini_config_data['ASSISTANT_MODEL'] = input("Enter Gemini Assistant Model Name (e.g., gemini-1.5-pro-latest): ").strip()
            if embedding_api_provider == "gemini":
                gemini_config_data['EMBEDDING_MODEL'] = input("Enter Gemini Embedding Model Name (e.g., models/embedding-001): ").strip()
            config['GEMINI'] = gemini_config_data
        
        with open(config_file_path, 'w') as configfile:
            config.write(configfile)
        print(f"\nConfiguration saved to {config_file_path}")
    else:
        config.read(config_file_path)

    # Validation of existing config
    if not config.has_section('GENERAL') or \
       'ASSISTANT_API_PROVIDER' not in config['GENERAL'] or \
       'EMBEDDING_API_PROVIDER' not in config['GENERAL']:
        print(f"GENERAL configuration in {config_file_path} is incomplete. Please delete it and re-run to configure.")
        sys.exit(1)

    assistant_provider = config['GENERAL']['ASSISTANT_API_PROVIDER']
    embedding_provider = config['GENERAL']['EMBEDDING_API_PROVIDER']

    if assistant_provider == "openai":
        if not config.has_section('OPENAI') or \
           'API_KEY' not in config['OPENAI'] or \
           'ASSISTANT_MODEL' not in config['OPENAI']:
            print(f"OpenAI assistant configuration in {config_file_path} is incomplete. Delete and re-run.")
            sys.exit(1)
    elif assistant_provider == "gemini":
        if not LANGCHAIN_GOOGLE_GENAI_AVAILABLE:
            print("Error: Configured for Gemini Assistant, but 'langchain-google-genai' is not installed.")
            sys.exit(1)
        if not config.has_section('GEMINI') or \
           'API_KEY' not in config['GEMINI'] or \
           'ASSISTANT_MODEL' not in config['GEMINI']:
            print(f"Gemini assistant configuration in {config_file_path} is incomplete. Delete and re-run.")
            sys.exit(1)
    
    if embedding_provider == "openai":
        if not config.has_section('OPENAI') or \
           'API_KEY' not in config['OPENAI'] or \
           'EMBEDDING_MODEL' not in config['OPENAI']:
            print(f"OpenAI embedding configuration in {config_file_path} is incomplete. Delete and re-run.")
            sys.exit(1)
    elif embedding_provider == "gemini":
        if not LANGCHAIN_GOOGLE_GENAI_AVAILABLE:
            print("Error: Configured for Gemini Embeddings, but 'langchain-google-genai' is not installed.")
            sys.exit(1)
        if not config.has_section('GEMINI') or \
           'API_KEY' not in config['GEMINI'] or \
           'EMBEDDING_MODEL' not in config['GEMINI']:
            print(f"Gemini embedding configuration in {config_file_path} is incomplete. Delete and re-run.")
            sys.exit(1)
            
    return config

def load_config(git_root):
    return ensure_askgit_dir_and_config(git_root)

# --- LLM and Embeddings Initializers ---
def get_llm(config, temperature=0.2):
    provider = config['GENERAL']['ASSISTANT_API_PROVIDER']
    if provider == "openai":
        return ChatOpenAI(
            openai_api_key=config['OPENAI']['API_KEY'],
            model_name=config['OPENAI']['ASSISTANT_MODEL'],
            openai_api_base=config['OPENAI'].get('BASE_URL') if config['OPENAI'].get('BASE_URL') else None,
            temperature=temperature
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=config['GEMINI']['ASSISTANT_MODEL'],
            google_api_key=config['GEMINI']['API_KEY'],
            temperature=temperature
        )
    else:
        raise ValueError(f"Unsupported Assistant API provider: {provider}")

def get_embeddings_model(config):
    provider = config['GENERAL']['EMBEDDING_API_PROVIDER']
    if provider == "openai":
        return OpenAIEmbeddings(
            openai_api_key=config['OPENAI']['API_KEY'],
            model=config['OPENAI']['EMBEDDING_MODEL'],
            openai_api_base=config['OPENAI'].get('BASE_URL') if config['OPENAI'].get('BASE_URL') else None,
        )
    elif provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=config['GEMINI']['EMBEDDING_MODEL'],
            google_api_key=config['GEMINI']['API_KEY']
        )
    else:
        raise ValueError(f"Unsupported Embedding API provider: {provider}")

# --- Git Utilities ---
def is_git_repository():
    return get_git_root() is not None

def get_git_tracked_files(git_root):
    try:
        files_blob = subprocess.check_output(
            ['git', '-C', str(git_root), 'ls-files', '-z'],
            stderr=subprocess.PIPE
        )
        files = files_blob.decode('utf-8').strip('\x00').split('\x00')
        return [git_root / f for f in files if f] 
    except subprocess.CalledProcessError as e:
        print(f"Error listing git files: {e.stderr.decode()}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"An unexpected error occurred while listing git files: {e}", file=sys.stderr)
        return []

# --- Whole Project Context Builder ---
def build_whole_project_context(git_root, max_chars=MAX_CONTEXT_CHARS):
    print(f"Building whole project context (max {max_chars} chars). This may take a while and consume many tokens...")
    context_lines = []
    tracked_files = get_git_tracked_files(git_root)
    total_chars = 0
    files_processed_count = 0
    files_skipped_count = 0

    for file_path_abs in tqdm(tracked_files, desc="Reading files for whole context"):
        file_path_rel = file_path_abs.relative_to(git_root)
        try:
            if file_path_abs.suffix.lower() in BINARY_LIKE_EXTENSIONS:
                files_skipped_count += 1
                continue
            
            with open(file_path_abs, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                files_skipped_count += 1
                continue

            file_header = f"$$FILE {str(file_path_rel)}\n"
            file_data = file_header + content.strip() + "\n\n"
            
            if max_chars is not None and (total_chars + len(file_data)) > max_chars:
                print(f"\nWarning: Whole project context reached character limit ({max_chars}). Truncating.")
                print(f"Processed {files_processed_count} files before truncation. Skipped {files_skipped_count} binary/empty files.")
                remaining_chars = max_chars - total_chars
                if remaining_chars > len(file_header):
                    context_lines.append(file_data[:remaining_chars])
                elif remaining_chars > 0:
                     context_lines.append(file_header[:remaining_chars])
                total_chars = max_chars 
                break 
            
            context_lines.append(file_data)
            total_chars += len(file_data)
            files_processed_count +=1

        except Exception as e:
            print(f"Skipping file {file_path_rel} for whole context due to error: {e}", file=sys.stderr)
            files_skipped_count +=1
    
    if not context_lines:
        print("No text content found to build project context.")
        return None
    
    full_context = "".join(context_lines)
    print(f"\nWhole project context built. Total characters: {total_chars} from {files_processed_count} files.")
    print(f"(Skipped {files_skipped_count} binary, empty, or error-prone files)")
    
    estimated_tokens = total_chars / 3.5 
    print(f"Estimated tokens (very approximate): {estimated_tokens:.0f}")
    if estimated_tokens > 100000:
         print("WARNING: The generated context is very large and will likely exceed LLM token limits or be very expensive!")
    elif total_chars == max_chars:
        print("INFO: Context was truncated to fit the character limit.")
    return full_context

# --- Core Logic: Scan ---
def scan_repository(config, git_root):
    embedding_provider = config['GENERAL']['EMBEDDING_API_PROVIDER']
    print(f"Scanning repository (RAG mode) using {embedding_provider} embeddings...")
    askgit_dir = get_askgit_dir(git_root)
    db_path_obj = get_db_path(askgit_dir)

    try:
        embeddings = get_embeddings_model(config)
    except Exception as e:
        print(f"Error initializing Embeddings model ({embedding_provider}): {e}")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    all_docs = []
    tracked_files = get_git_tracked_files(git_root)
    
    if not tracked_files:
        print("No files found to scan.")
        return

    print(f"Found {len(tracked_files)} files to process for RAG.")
    files_processed_count = 0
    files_skipped_count = 0

    for file_path_abs in tqdm(tracked_files, desc="Processing files for RAG"):
        file_path_rel = file_path_abs.relative_to(git_root)
        try:
            if file_path_abs.suffix.lower() in BINARY_LIKE_EXTENSIONS:
                files_skipped_count += 1
                continue
            
            with open(file_path_abs, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                files_skipped_count += 1
                continue

            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={"source": str(file_path_rel), "chunk": i}
                )
                all_docs.append(doc)
            files_processed_count +=1
        except Exception as e:
            print(f"Skipping file {file_path_rel} during scan due to error: {e}", file=sys.stderr)
            files_skipped_count +=1

    if not all_docs:
        print("No text content found to create a RAG knowledge base.")
        print(f"(Processed {files_processed_count} files, skipped {files_skipped_count} binary/empty/error files)")
        return

    print(f"\nGenerated {len(all_docs)} document chunks from {files_processed_count} files. Creating RAG vector store...")
    print(f"(Skipped {files_skipped_count} binary, empty, or error-prone files during scan)")
    try:
        vector_store = FAISS.from_documents(all_docs, embeddings)
        vector_store.save_local(folder_path=str(db_path_obj.parent), index_name=db_path_obj.name)
        print(f"RAG Knowledge base saved to {db_path_obj.parent / (db_path_obj.name + '.faiss')}")
    except Exception as e:
        print(f"Error creating or saving FAISS index: {e}")

# --- Core Logic: Ask (RAG Mode) ---
def ask_question_rag(config, question, git_root, suppress_print=False):
    askgit_dir = get_askgit_dir(git_root)
    db_path_obj = get_db_path(askgit_dir)
    assistant_provider = config['GENERAL']['ASSISTANT_API_PROVIDER']
    embedding_provider = config['GENERAL']['EMBEDDING_API_PROVIDER']


    if not (db_path_obj.parent / (db_path_obj.name + ".faiss")).exists():
        if not suppress_print:
            print("RAG Knowledge base not found. Please run 'scan' first.")
        return None

    try:
        embeddings = get_embeddings_model(config) # Uses EMBEDDING_API_PROVIDER
        llm = get_llm(config, temperature=0.2)    # Uses ASSISTANT_API_PROVIDER
    except Exception as e:
        if not suppress_print:
            print(f"Error initializing models (Assistant: {assistant_provider}, Embedding: {embedding_provider}) for RAG: {e}")
        return None

    try:
        vector_store = FAISS.load_local(
            folder_path=str(db_path_obj.parent), 
            embeddings=embeddings, 
            index_name=db_path_obj.name,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        if not suppress_print:
            print(f"Error loading FAISS index for RAG (Embeddings: {embedding_provider}): {e}")
        return None

    retriever = vector_store.as_retriever(search_kwargs={"k": 7})

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise. Be sure to cite the source file(s) from the context if used in your answer, like (Source: path/to/file.py)."),
        ("user", "Question: {input}\n\nContext:\n{context}")
    ])
    
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    if not suppress_print:
        print(f"Asking LLM (Assistant: {assistant_provider}, Embeddings: {embedding_provider}, RAG mode)...")
    try:
        response = retrieval_chain.invoke({"input": question})
        answer = response["answer"]
        
        if not suppress_print:
            print("\nAnswer (RAG):")
            print(answer)
        return answer
    except Exception as e:
        message = f"Error during RAG LLM call (Assistant: {assistant_provider}): {e}"
        if not suppress_print:
            print(message)
        else:
            print(message + " (called from doc generation)")
        return None

# --- Core Logic: Ask (Whole Project Mode) ---
def ask_question_whole(config, question, git_root, suppress_print=False):
    assistant_provider = config['GENERAL']['ASSISTANT_API_PROVIDER']
    if not suppress_print:
        print(f"Using whole project context for 'ask' command (Assistant: {assistant_provider})...")
    
    project_context = build_whole_project_context(git_root)
    if not project_context:
        if not suppress_print:
            print("Could not build project context. No files or an error occurred.")
        return None

    try:
        llm = get_llm(config, temperature=0.2) # Uses ASSISTANT_API_PROVIDER
    except Exception as e:
        if not suppress_print:
            print(f"Error initializing LLM (Assistant: {assistant_provider}) for whole mode: {e}")
        return None

    system_prompt_content = (
        "You are an assistant for question-answering tasks. The entire project's content is provided below, "
        "with each file prefixed by '$$FILE path/to/file'. Use this content to answer the user's question. "
        "If you don't know the answer from the provided content, say that you don't know. Keep the answer concise.\n\n"
        "--- PROJECT CONTENT START ---\n{project_context}\n--- PROJECT CONTENT END ---"
    )
    
    messages = [
        SystemMessage(content=system_prompt_content.format(project_context=project_context)),
        HumanMessage(content=f"Question: {question}")
    ]

    if not suppress_print:
        print(f"Asking LLM (Assistant: {assistant_provider}, whole project mode)...")
    try:
        response = llm.invoke(messages)
        answer = response.content
        if not suppress_print:
            print("\nAnswer (Whole Project):")
            print(answer)
        return answer
    except Exception as e:
        message = f"Error during whole project LLM call (Assistant: {assistant_provider}): {e}"
        if "maximum context length" in str(e).lower() or "context_length_exceeded" in str(e).lower() or ("token" in str(e).lower() and "limit" in str(e).lower()):
            message += "\nThe project context likely exceeded the model's maximum token limit. Consider using RAG mode (without --whole) or try with a smaller project/different model."
        if not suppress_print:
            print(message)
        else:
             print(message + " (called from doc generation)")
        return None

# --- Dispatcher for Ask ---
def ask_command_func(config, question, git_root, whole_mode=False):
    if whole_mode:
        ask_question_whole(config, question, git_root)
    else:
        ask_question_rag(config, question, git_root)

# --- Core Logic: Research (RAG Mode) ---
def research_mode_rag(config, git_root):
    askgit_dir = get_askgit_dir(git_root)
    db_path_obj = get_db_path(askgit_dir)
    assistant_provider = config['GENERAL']['ASSISTANT_API_PROVIDER']
    embedding_provider = config['GENERAL']['EMBEDDING_API_PROVIDER']

    if not (db_path_obj.parent / (db_path_obj.name + ".faiss")).exists():
        print("RAG Knowledge base not found. Please run 'scan' first.")
        return

    try:
        embeddings = get_embeddings_model(config)
        llm = get_llm(config, temperature=0.3)
    except Exception as e:
        print(f"Error initializing models (Assistant: {assistant_provider}, Embedding: {embedding_provider}) for RAG research: {e}")
        return

    try:
        vector_store = FAISS.load_local(
            folder_path=str(db_path_obj.parent), 
            embeddings=embeddings, 
            index_name=db_path_obj.name,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Error loading FAISS index for RAG research (Embeddings: {embedding_provider}): {e}")
        return

    retriever = vector_store.as_retriever(search_kwargs={"k": 7})

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, just say that "
        "you don't know. Keep the answer concise. "
        "Cite source file(s) when appropriate, like (Source: path/to/file.py)."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    chat_history_for_chain = InMemoryChatMessageHistory()
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda _: chat_history_for_chain,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    print(f"Entering research mode (RAG, Assistant: {assistant_provider}, Embeddings: {embedding_provider}). Type 'exit' or 'quit' to end.")
    session_id = f"askgit_research_rag_{assistant_provider}_{embedding_provider}_session"

    while True:
        try:
            user_input = input(f"\nYou (RAG - A:{assistant_provider}, E:{embedding_provider}): ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue

            print(f"AI (RAG - A:{assistant_provider}): Thinking...")
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            print(f"AI (RAG - A:{assistant_provider}): {response['answer']}")
        except KeyboardInterrupt:
            print("\nExiting RAG research mode.")
            break
        except Exception as e:
            print(f"An error occurred in RAG research (Assistant: {assistant_provider}): {e}")

# --- Core Logic: Research (Whole Project Mode) ---
def research_mode_whole(config, git_root):
    assistant_provider = config['GENERAL']['ASSISTANT_API_PROVIDER']
    print(f"Entering research mode (whole project, Assistant: {assistant_provider}). This might be slow/expensive.")
    project_context = build_whole_project_context(git_root)

    if not project_context:
        print("Could not build project context for whole project research. Exiting.")
        return

    try:
        llm = get_llm(config, temperature=0.3) # Uses ASSISTANT_API_PROVIDER
    except Exception as e:
        print(f"Error initializing LLM (Assistant: {assistant_provider}) for whole project research: {e}")
        return

    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "The entire project's content is provided below, with each file prefixed by '$$FILE path/to/file'. "
        "Use this content AND the chat history to answer the user's questions. "
        "If you don't know the answer from the provided content and history, say that you don't know. "
        "Keep answers concise.\n\n"
        "--- PROJECT CONTENT START ---\n"
        f"{project_context}\n"
        "--- PROJECT CONTENT END ---"
    )
    
    if len(system_message_content) / 3.5 > 120000: 
        print("WARNING: The initial project context for research mode is extremely large.")
        print("This may lead to errors, very slow responses, or high costs. Consider RAG mode for large projects.")

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message_content),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}")
    ])
    
    chain = prompt_template | llm
    chat_history_for_chain = InMemoryChatMessageHistory()

    conversational_chain = RunnableWithMessageHistory(
        chain,
        lambda _: chat_history_for_chain,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    print(f"Research mode (whole project - Assistant: {assistant_provider}) active. Type 'exit' or 'quit' to end.")
    session_id = f"askgit_research_whole_{assistant_provider}_session"

    while True:
        try:
            user_input = input(f"\nYou (Whole - A:{assistant_provider}): ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue

            print(f"AI (Whole - A:{assistant_provider}): Thinking...")
            ai_response_message = conversational_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            print(f"AI (Whole - A:{assistant_provider}): {ai_response_message.content}")

        except KeyboardInterrupt:
            print("\nExiting whole project research mode.")
            break
        except Exception as e:
            print(f"An error occurred in whole project research (Assistant: {assistant_provider}): {e}")
            if "maximum context length" in str(e).lower() or "context_length_exceeded" in str(e).lower() or ("token" in str(e).lower() and "limit" in str(e).lower()):
                print("The conversation history combined with project context likely exceeded the model's maximum token limit.")
                print("Try starting a new research session or asking shorter questions.")

# --- Dispatcher for Research ---
def research_command_func(config, git_root, whole_mode=False):
    if whole_mode:
        research_mode_whole(config, git_root)
    else:
        research_mode_rag(config, git_root)

# --- Core Logic: Doc Command ---
def doc_command_func(config, output_filename_base, question, git_root, whole_mode=False):
    askgit_dir = get_askgit_dir(git_root)
    docs_output_dir = get_docs_dir(askgit_dir)
    assistant_provider = config['GENERAL']['ASSISTANT_API_PROVIDER']
    embedding_provider = config['GENERAL']['EMBEDDING_API_PROVIDER'] # Relevant if not --whole
    
    safe_filename_base = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in output_filename_base)
    if not safe_filename_base:
        safe_filename_base = "askgit_doc"
    output_file_path = docs_output_dir / f"{safe_filename_base}.md"

    mode_str = "whole project" if whole_mode else f"RAG (Embeddings: {embedding_provider})"
    print(f"Generating document for question: \"{question}\" (Assistant: {assistant_provider}, Mode: {mode_str})")
    print(f"Output will be saved to: {output_file_path}")

    answer = None
    if whole_mode:
        answer = ask_question_whole(config, question, git_root, suppress_print=True)
    else:
        answer = ask_question_rag(config, question, git_root, suppress_print=True)

    if answer:
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Query: {question}\n\n")
                f.write(f"## Answer (Generated by Assistant: {assistant_provider}, Mode: {mode_str}):\n\n")
                f.write(answer.strip() + "\n")
            print(f"Document saved successfully to {output_file_path}")
        except IOError as e:
            print(f"Error saving document: {e}")
    else:
        print("No answer received from LLM. Document not created.")

# --- Main Execution ---
def main():
    if not is_git_repository():
        print("Error: This command must be run inside a git repository.", file=sys.stderr)
        sys.exit(1)

    git_root = get_git_root()
    config = load_config(git_root) 

    parser = argparse.ArgumentParser(description="AskGit: Query your git repository using LLMs with configurable providers.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    parser_scan = subparsers.add_parser("scan", help="Scan repository and build/update RAG knowledge base.")
    parser_scan.set_defaults(func=lambda args: scan_repository(config, git_root))

    parser_ask = subparsers.add_parser("ask", help="Ask a single question.")
    parser_ask.add_argument("question", type=str, help="The question to ask.")
    parser_ask.add_argument("--whole", action="store_true", help="Use whole project context (experimental, high token usage) instead of RAG.")
    parser_ask.set_defaults(func=lambda args: ask_command_func(config, args.question, git_root, args.whole))

    parser_research = subparsers.add_parser("research", help="Enter multi-turn research mode.")
    parser_research.add_argument("--whole", action="store_true", help="Use whole project context (experimental, high token usage) instead of RAG.")
    parser_research.set_defaults(func=lambda args: research_command_func(config, git_root, args.whole))
    
    parser_doc = subparsers.add_parser("doc", help="Generate a document (.md) from a question's answer.")
    parser_doc.add_argument("output_filename_base", type=str, help="Base name for output .md file. Saved in .askgit/docs/")
    parser_doc.add_argument("question", type=str, help="The question for the document content.")
    parser_doc.add_argument("--whole", action="store_true", help="Use whole project context (experimental, high token usage) instead of RAG.")
    parser_doc.set_defaults(func=lambda args: doc_command_func(config, args.output_filename_base, args.question, git_root, args.whole))
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()