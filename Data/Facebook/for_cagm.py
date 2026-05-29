with open("x.txt", "r") as infile, open("y.txt", "w") as outfile:
    for lineno, line in enumerate(infile, start=1):
        numbers = line.strip().split()
        ints = [str(int(float(num))) for num in numbers]
        outfile.write(f"{lineno} {' '.join(ints)}\n")
