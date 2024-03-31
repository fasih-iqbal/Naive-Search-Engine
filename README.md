# Naive-Search-Engine
Developing a Naïve Search Engine Utilising MapReduce Technology

### Overview
This project aims to calculate the relevance of documents based on a given query text. The system utilizes techniques such as preprocessing, TF-IDF (Term Frequency-Inverse Document Frequency), and vector space model to determine the relevance of documents to a user's query.

### Features
1. **Preprocessing**: The text data undergoes preprocessing, including tokenization, stop word removal, and lemmatization to enhance the quality of analysis.
   
2. **TF-IDF Calculation**: The project calculates TF-IDF weights for each term in the corpus. This step involves computing Term Frequency (TF), Document Frequency (DF), and finally TF-IDF weights.

3. **Vector Space Model**: The system implements a vector space model to represent documents and queries as vectors in a high-dimensional space. This model facilitates the comparison of documents based on their relevance to the query.

4. **Relevance Calculation**: Relevance between the query vector and each document vector is computed using both sparse and full representations. This step involves calculating the cosine similarity between vectors to determine document relevance.

### File Structure
- **mapper.py**: Performs pre-processing and all calculations related to TF-IDF and vector space model.
- **reducer.py**: Takes the results from the mapper file and reduces them to display the top 10 relevant documents.
- **task.ipynb**: Jupyter notebook containing the main code, meeting all necessary requirements.

### Workflow
1. **Preprocessing**: The text data is preprocessed to remove noise and irrelevant information.
2. **TF-IDF Calculation**: Term Frequency (TF) and Inverse Document Frequency (IDF) are computed to obtain TF-IDF weights.
3. **Vector Space Model Implementation**: Documents and queries are represented as vectors in a high-dimensional space.
4. **Relevance Calculation**: Relevance between the query and each document is calculated using cosine similarity.
5. **Result Display**: The top 10 relevant documents are displayed to the user.

### Dependencies
- Python 3
- pandas
- nltk
- numpy
- Apacha Hadoop

### Usage
1. Ensure all dependencies are installed.
2. Run `task.ipynb` to execute the code.
3. Enter the query text when prompted.
4. View the top 10 relevant documents displayed.

### Contributors
- **Fasih Iqbal**:  developer
- **Jawad Ahmad**:  developer
- **Hassan Rizwan**: developer

### Acknowledgments
- Special thanks to the contributors and reviewers for their valuable insights and feedback.
