import os
import random
import re
import warnings
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Setup and Helper Functions ---

def setup_environment():
    """Loads environment variables and sets up the API key."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file or environment variables.")
    os.environ["OPENAI_API_KEY"] = api_key

def load_pdf(path):
    """Loads a PDF file and splits it into pages."""
    loader = PyPDFLoader(path)
    return loader.load_and_split()

def build_vector_store(pages):
    """Creates a vector store from document pages."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store, docs

# --- Content Generation Functions ---

def generate_detailed_summary(llm, docs):
    """Generates a detailed summary from sampled document chunks."""
    chunks = len(docs)
    # Ensure at least one chunk is sampled, even for very small docs
    sample_indices = sorted(list(set([
        0,
        chunks // 3,
        (2 * chunks) // 3,
        max(0, chunks - 1)
    ])))
    
    sampled_content = "\n---\n".join([docs[i].page_content for i in sample_indices])

    prompt = (
        "Generate a detailed 1-page summary of the provided text, which consists of key sections from a chapter. "
        "Synthesize these sections to create a coherent overview that covers the main concepts, "
        "definitions, and examples from beginning to end.\n\n"
        "Here is the content:\n"
        + sampled_content
    )
    return llm.invoke(prompt).content

def generate_short_notes(llm, docs):
    """Generates concise, bullet-point style notes."""
    full_text = "\n".join([doc.page_content for doc in docs])
    prompt = (
        "Generate very short, concise notes in bullet-point format from the following text. "
        "Focus on key terms, definitions, and main ideas.\n\n"
        "Text:\n" + full_text[:4000] # Increased character limit for better context
    )
    return llm.invoke(prompt).content

def generate_mcq_text(llm, docs, num_q):
    """Generates the raw text for multiple-choice questions."""
    text = "\n".join([doc.page_content for doc in docs])[:4000] # Increased limit
    prompt = (
        f"Create exactly {num_q} multiple-choice questions from the provided text. "
        "STRICTLY follow this format for each question:\n"
        "<question number>. <question text>\n"
        "A) <option 1>\n"
        "B) <option 2>\n"
        "C) <option 3>\n"
        "D) <option 4>\n"
        "Correct Answer: <The correct letter, e.g., B>\n\n"
        "Here is the text:\n\n"
        + text
    )
    return llm.invoke(prompt).content

def generate_long_questions(llm, pages, num_q):
    """Generates long-form questions based on random pages."""
    total_pages = len(pages)
    if total_pages == 0:
        return "No pages found in the document to generate questions from."

    # Select unique random pages
    num_to_sample = min(num_q, total_pages)
    selected_indices = random.sample(range(total_pages), num_to_sample)
    
    questions = []
    for i, idx in enumerate(selected_indices):
        page_content = pages[idx].page_content
        prompt = (
            f"Generate 1 insightful, long-answer question that requires a detailed explanation, "
            f"based on the following page content.\n"
            f"Format the output as: Q{i+1}. <question> (Source: Page {idx+1})\n\n"
            "Page Content:\n"
            + page_content
        )
        question = llm.invoke(prompt).content.strip()
        questions.append(question)

    return "\n\n".join(questions)

# --- New Interactive Quiz Functions ---

def parse_mcqs(mcq_text):
    """
    Parses raw MCQ text into a list of question dictionaries.
    This version uses a more flexible regex to handle different numbering formats.
    """
    # ** THE FIX IS HERE **
    # This regex splits the text before a line that starts with a number, an optional 'Q',
    # and a dot or colon. This handles "1.", "Q1.", "Question 1:", etc.
    question_blocks = re.split(r'\n(?=(?:Q|Question)?\s*\d+[\.:])', mcq_text.strip())
    
    parsed_questions = []
    for block in question_blocks:
        if not block.strip():
            continue
            
        try:
            # Extract the correct answer first
            answer_match = re.search(r"Correct Answer:\s*([A-D])", block, re.IGNORECASE)
            if not answer_match:
                continue

            correct_answer = answer_match.group(1).upper()
            
            # The question text is everything before the "Correct Answer:" line
            question_content = block[:answer_match.start()].strip()

            parsed_questions.append({
                "question_text": question_content,
                "correct_answer": correct_answer
            })
        except (AttributeError, IndexError):
            # Skip blocks that are not formatted correctly
            print(f"\nWarning: Skipping a malformed MCQ block:\n---\n{block}\n---")
            continue
            
    return parsed_questions


def run_mcq_quiz(llm, docs, num_q):
    """Generates, presents, and scores an interactive MCQ quiz."""
    print(f"\n--- PART 3: Interactive Quiz ({num_q} MCQs) ---")
    
    if num_q == 0:
        print("Skipping quiz as requested.")
        return

    print("\nGenerating questions, please wait...")
    mcq_raw_text = generate_mcq_text(llm, docs, num_q)
    parsed_questions = parse_mcqs(mcq_raw_text)

    if not parsed_questions:
        print("\nError: Could not parse any questions from the model's output.")
        print("This might be due to an unusual format. Here is the raw output received:")
        print("--------------------")
        print(mcq_raw_text)
        print("--------------------")
        return

    score = 0
    # Use the number of successfully parsed questions for the quiz
    total_questions = len(parsed_questions)
    
    for i, q_data in enumerate(parsed_questions):
        print("\n" + "="*20)
        print(f"Question {i+1}/{total_questions}")
        print(q_data["question_text"])
        
        user_answer = ""
        while user_answer not in ['A', 'B', 'C', 'D']:
            user_answer = input("Your answer (A, B, C, or D): ").upper().strip()
            if user_answer not in ['A', 'B', 'C', 'D']:
                print("Invalid input. Please enter A, B, C, or D.")

        if user_answer == q_data["correct_answer"]:
            print(f"✅ Correct! The answer is {q_data['correct_answer']}.")
            score += 1
        else:
            print(f"❌ Incorrect. The correct answer was {q_data['correct_answer']}.")

    print("\n" + "="*20)
    print("🎉 Quiz Complete! 🎉")
    if total_questions > 0:
      print(f"Your final score is: {score} out of {total_questions} ({score/total_questions:.2%})")
    else:
      print("No questions were asked.")
    print("="*20)


# --- Main Agent Function ---

def revision_agent(pdf_path):
    """Main function to orchestrate the revision guide generation."""
    print("🚀 Starting the Student Reviser Guide AI Agent...")
    
    try:
        setup_environment()
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    print(f"\n📚 Loading and processing PDF: {pdf_path}")
    pages = load_pdf(pdf_path)
    _, docs = build_vector_store(pages)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    # --- User Inputs ---
    try:
        mcq_count = int(input("How many MCQ questions do you want for the quiz? (Enter 0 to skip) "))
        longq_count = int(input("How many long-answer questions do you want? (Enter 0 to skip) "))
    except ValueError:
        print("Invalid input. Please enter a number. Exiting.")
        return

    # --- Generate and Display Content ---
    print("\n--- PART 1: Detailed Summary Notes ---")
    print(generate_detailed_summary(llm, docs))

    print("\n--- PART 2: Very Short Notes ---")
    print(generate_short_notes(llm, docs))

    # Run the interactive quiz
    run_mcq_quiz(llm, docs, mcq_count)

    if longq_count > 0:
        print(f"\n--- PART 4: {longq_count} Long Questions with Page Numbers ---")
        print(generate_long_questions(llm, pages, longq_count))

    print("\n✅ All tasks complete. Happy revising!")


if __name__ == "__main__":
    # IMPORTANT: Replace this with the actual path to your PDF file.
    # Using a raw string (r"...") or forward slashes is recommended for path compatibility.
    pdf_file_path = r"E:\Student Reviser Guide AI Agent\Binary Trees.pdf" 
    
    if not os.path.exists(pdf_file_path):
        print(f"Error: The file was not found at the specified path: {pdf_file_path}")
        print("Please update the 'pdf_file_path' variable in the code with the correct location of your PDF.")
    else:
        revision_agent(pdf_file_path)