from dotenv import load_dotenv
from app import create_interface

if __name__ == "__main__":
    load_dotenv()
    demo = create_interface()
    demo.launch(share=False)
