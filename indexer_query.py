import sys
import os

from mapper import preprocess_text
from reducer import rank_documents, print_ranked_documents


def run_indexer(input_files, output_file):
    # Assuming subset.csv is located in the Downloads directory
    input_files = "/home/your_username/Downloads/subset.csv"

    # Your Hadoop command to run the Indexer job
    hadoop_command = f"hadoop jar /path/to/jar Indexer {
        input_files} {output_file}"
    # Execute the command
    os.system(hadoop_command)


def execute_query(output_length, query):
    # Your Hadoop command to run the Query job
    hadoop_command = f"hadoop jar /path/to/jar Query {
        output_length} \"{query}\""
    # Execute the command
    os.system(hadoop_command)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python main_script.py <output_file> <output_length> <query>")
        sys.exit(1)

    # Output file where Indexer job will store its results
    output_file = sys.argv[1]

    # Length of the output list of relevant documents
    output_length = sys.argv[2]

    # Query text
    query = " ".join(sys.argv[3:])

    # Run Indexer
    run_indexer(output_file)

    # Execute Query
    execute_query(output_length, query)
