import argparse
from utils.worddetector import detect_words

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WordDetector')
    parser.add_argument("path")
    args = parser.parse_args()
    path = args.path.replace('\'', '')
    path = path.replace('\"','')
    detect_words(path)
