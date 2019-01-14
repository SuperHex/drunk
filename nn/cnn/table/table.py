PATH = "/home/cirno/Downloads/114514/data.csv"

def identity(x):
    return x

def parse_table(path):
    f = open(PATH, 'r')
    lines = f.readlines()
    lines = [line.strip('\t\r\n').split(',') for line in lines[1:]]
    f.close()
    return lines

def get_table_data(lines, f=identity):
    d = {}
    for line in lines:
        d[int(line[0])] = f(line)
    return d
