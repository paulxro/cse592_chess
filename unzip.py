import zstandard as zstd
import sys

def unzip_zst(input_file, output_file):
    with open(input_file, 'rb') as infile, open(output_file, 'wb') as outfile:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(infile) as reader:
            outfile.write(reader.read())

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    unzip_zst(input_file, output_file)