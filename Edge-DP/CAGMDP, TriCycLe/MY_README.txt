# MY README

- You only need to evaluate DPCAGM (1) and TriCycLe (2).
- Usage: java -jar NewDPC.jar <DATANAME> <NUM_TRIALS> [1|2]
    - 1: DPCAGM Method
    - 2: DPTriCycLe Method

- Expected files in same directory:
    - attribute_<DATANAME>_combined.txt: Node feature matrix, with row elements separated by space.
        - The FIRST element of each row is the node_id.
    - <DATANAME>_combined.txt: Edge list

- Refer to the other README to understand the output files.

- Things to note:
    - In the feature matrix, only the first 50 features will be used. Change this by changing 
      line 139 in DPCAGMGenerator.java and line 348 in DPTriCycle.java. The variable is: int num_abt.

    - To compile:
        - Create a ./bin directory and make sure it's empty.
        - ENSURE THAT THERE ARE NO HIDDEN .IPYNB CHECKPOINTS ETC. LYING AROUND!
        - find src -name "*.java" > sources.txt (DON'T NEED TO DO THIS IF SOURCES ALREADY EXISTS).
        - javac -d bin -cp "lib/*" @sources.txt
        - If not already there, manifest.txt should be like:
            Manifest-Version: 1.0
            Main-Class: ExperimentMain
            Class-Path: 
            - Then, append to it: for jar in lib/*.jar; do echo -n "$jar " >> manifest.txt; done; echo "" >> manifest.txt
        - jar cfm NewDPC.jar manifest.txt -C bin .

    - If you get a FileNotFound exception, you MAY need to create the folders yourself to avoid errors.
        
