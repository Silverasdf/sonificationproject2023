# Human Test.py - can do three different kinds of tests for humans to see if they can tell the difference between audio files
# Ryan Peruski, 07/25/23
# usage: python3 HumanTest.py <1, 2, or 3> <directory of .wav files> <output directory>
# Different wav files will be in your output directory. Listen to them and then enter the correct answer as shown on screen.
# Output is a csv file with the results of the test in the output directory

import sys, os
import pandas as pd
import random

def test(test_number, wav_dir, output_dir):
    #Make output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    #Get list of all .wav files in wav_dir
    combined_wav_files = [file for file in os.listdir(wav_dir + '/combined_on_wavs') if file.endswith('.wav')]
    on_wav_files = [file for file in os.listdir(wav_dir + '/on_wavs') if file.endswith('.wav')]
    off_wav_files = [file for file in os.listdir(wav_dir + '/off_wavs') if file.endswith('.wav')]
    # If any of these are empty, then exit
    if len(combined_wav_files) == 0 or len(on_wav_files) == 0 or len(off_wav_files) == 0:
        print("Directory must have .wav files", file=sys.stderr)
        sys.exit(1)

    truth = ''
    answer = ''
    file_A = ''
    file_B = ''
    file_C = ''

    # Test one provides two samples from the off_wav_files A and B (A can equal B)
    # Then, the user is asked whether A = B or not

    # Test two provides two samples from the on_wav_files A and B (A can equal B)
    # Then, the user is asked whether A = B or not
    if test_number == '1' or test_number == '2':
        if test_number == '1':
            wav_list = off_wav_files
            string = '/off_wavs/'
        else:
            wav_list = on_wav_files
            string = '/on_wavs/'
        #Get two random files from off_wav_files
        file_A = random.choice(wav_list)
        #Remove A from off_wav_files
        wav_list.remove(file_A)
        
        # 50-50 chance of A = B or A != B
        truth = random.choice(['A', 'B'])
        if truth == 'A':
            #A = B
            file_B = file_A
        else:
            #A != B
            file_B = random.choice(wav_list)
        #Copy and rename files to output_dir:
        os.system('cp ' + wav_dir + string + file_A + ' ' + output_dir + '/A.wav')
        os.system('cp ' + wav_dir + string + file_B + ' ' + output_dir + '/B.wav')

        print("Enter the correct answer (A/B):")
        print("A. A = B")
        print("B. A != B")
        while(True):
            answer = input()
            if answer == 'A' or answer == 'B':
                break
            print("Invalid input. Enter A or B")

    # Test three provides two samples from the on_wav_files A and B (A cannot equal B)
    # The user is then provided a third sample C. C can be A, B, or A+B
    # Then, the user is asked whether C = A, C = B, or C = A+B
    if test_number == '3':
        wav_list = on_wav_files
        file_A = random.choice(wav_list)
        #Remove A from off_wav_files
        wav_list.remove(file_A)
        file_B = random.choice(wav_list)
        #Remove B from off_wav_files
        wav_list.remove(file_B)
        # 33-33-33 chance of C = A, C = B, or C = A+B
        truth = random.choice(['A', 'B', 'C'])
        if truth == 'A':
            #C = A
            file_C = file_A
        elif truth == 'B':
            #C = B
            file_C = file_B
        else:
            #C = A+B
            file_C = file_A[:-4] + "_" + file_B[:-4] + '.wav'
        #Copy and rename files to output_dir:
        os.system('cp ' + wav_dir + '/on_wavs/' + file_A + ' ' + output_dir + '/A.wav')
        os.system('cp ' + wav_dir + '/on_wavs/' + file_B + ' ' + output_dir + '/B.wav')
        os.system('cp ' + wav_dir + '/combined_on_wavs/' + file_C + ' ' + output_dir + '/C.wav')
        print("Enter the correct answer (A/B/C):")
        print("A. C = A")
        print("B. C = B")
        print("C. C = A+B")
        while(True):
            answer = input()
            if answer in ['A', 'B', 'C']:
                break
            print("Invalid input. Enter A, B, or C")

    df = pd.DataFrame(columns=['Test_Num', 'File_A', "File_B", "File_C", 'Truth', 'Answer']) 
    df.loc[len(df.index)] = [test_number, file_A, file_B, file_C, truth, answer]
    df.to_csv(output_dir + '/results.csv', index=False)
    print("Correct answer: " + truth)
    print("Your answer: " + answer)
    if truth == answer:
        print("Correct!")
    else:
        print("Incorrect!")
    

def main():
    #Parse arguments
    if len(sys.argv) != 4:
        print("usage: python3 HumanTest.py <1, 2, or 3> <directory of .wav files> <output directory>")
        sys.exit()
    test_number = sys.argv[1]
    wav_dir = sys.argv[2]
    output_dir = sys.argv[3]

    #If directory does not have combined_wavs, on_wavs, and off_wavs, then exit
    if not os.path.exists(wav_dir + '/combined_on_wavs') or not os.path.exists(wav_dir + '/on_wavs') or not os.path.exists(wav_dir + '/off_wavs'):
        print("Directory must have subdirectories combined_wavs, on_wavs, and off_wavs", file=sys.stderr)
        sys.exit(1)
    #If the test number is not 1, 2, or 3, then exit
    if test_number not in ['1', '2', '3']:
        #Print to stderr
        print("Test number must be 1, 2, or 3", file=sys.stderr)
        sys.exit(1)

    print("Starting test " + test_number + " on directory " + wav_dir + " with output directory " + output_dir)
    test(test_number, wav_dir, output_dir)

if __name__ == '__main__':
    main()