from usr_func.set_resume_state import set_resume_state
from usr_func.get_resume_state import get_resume_state
import os


if __name__ == "__main__":
    set_resume_state(False)
    print("resume:", get_resume_state())
    os.system("cat counter.txt")
    os.system("cat resume_flag.txt")

