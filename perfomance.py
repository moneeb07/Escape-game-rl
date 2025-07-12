import os
import sys
import subprocess

# Set up directory paths
ai_final_dir = r"D:\ai\ai_final"
project_dir = os.path.join(ai_final_dir, "The project itself")
src_dir = os.path.join(project_dir, "src")
level1_dir = os.path.join(ai_final_dir, "level1")
level3_path = os.path.join(ai_final_dir, "AI Proj", "level3.py")

# Add directories to Python path
sys.path.append(src_dir)
sys.path.append(project_dir)
sys.path.append(level1_dir)

# Change working directory to where the game assets are
os.chdir(project_dir)

def run_level1():
    os.chdir(level1_dir)
    result = subprocess.run([sys.executable, "run.py"], capture_output=True, text=True)
    os.chdir(project_dir)  # Change back to project directory
    return "Goal reached: True" in result.stdout

def run_level2():
    from src.game import main
    return main()

def run_level3():
    os.chdir(os.path.dirname(level3_path))
    subprocess.run([sys.executable, level3_path])

# run_level2()

if __name__ == "__main__":
    
    perfomance_level1 =17
    # for i in range(20):
    #     if run_level1():
    #         perfomance_level1 +=1
    #     print(i)
    print(f"level1 perfomance: {(perfomance_level1/20)*100}%")
    
    
    
    
#     perfomance_total =0
#     perfomance_level1 =0
#     perfomance_level2 =0
    
#     for i in range(20):
        
#         if run_level2():
#             perfomance_level2 +=1
        
#         if run_level1():
#             print("Level 1 completed successfully.")
#             perfomance_level1 +=1
#             if run_level2():
#                 print("Level 2 completed successfully.")
#                 check=run_level3()
#                 if check:
#                     perfomance_total +=1
#             else:
#                 print("Level 2 was not completed successfully.")
#         else:
#             print("Level 1 was not completed successfully.")



# print(f"game perfomance: {(perfomance_total/20)*100}%")
# print(f"level2 perfomance: {(perfomance_level2/20)*100}%")
