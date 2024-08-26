def rearrange_lines(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    first_line = lines[0]

    lines_01 = []
    lines_10 = []

    for line in lines[1:]:
        elements = line.strip().split(',')
        if elements[-2:] == ['0', '1']:
            lines_01.append(line)
        elif elements[-2:] == ['1', '0']:
            lines_10.append(line)

    with open('perf_labels.txt', 'w') as file:
        file.write(first_line)
        for line in lines_10:
            file.write(line)
        for line in lines_01:
            file.write(line)
       

rearrange_lines('a.txt')
