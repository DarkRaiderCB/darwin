from bs4 import BeautifulSoup as bs
from googlesearch import search
from rank_bm25 import BM25Okapi
import string
from sklearn.feature_extraction import _stop_words
from tqdm.autonotebook import tqdm
import numpy as np
import concurrent.futures
import time
import requests
import PyPDF2
import os
import re

from together import Together

client = Together()

def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc

def BM25func(passages, query):
    tokenized_corpus = []
    for passage in tqdm(passages):
        tokenized_corpus.append(bm25_tokenizer(passage))
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    print("BM25 SCORES:", len(bm25_scores))
    try:
        top_n = np.argpartition(bm25_scores, -10)[-10:]
    except:
        try:
            top_n = np.argpartition(bm25_scores, -4)[-4:]
        except:
            top_n = np.argpartition(bm25_scores, -2)[-2:]

    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
    bm25_passages = []
    for hit in bm25_hits:
        bm25_passages.append(' '.join(passages[hit["corpus_id"]].split()[:100]))
    print(bm25_passages)
    return bm25_passages

def scraper(url, con, DataWrtUrls, passages):
    print("Scrapper running")
    session = requests.Session()
    my_headers = {
        "User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 14685.0.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.4992.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9"
    }
    try:
        result = session.get(url, headers=my_headers, verify=False, timeout=3)
        doc = bs(result.content, "html.parser")
        contents = doc.find_all("p")
        for content in contents:
            passages.append(content.text)
            con.append(content.text + "\n")

        DataWrtUrls[url] = str(con)
    except Exception as exc:
        print(f"Error scraping {url}: {exc}")

def extract_links(URLs):
    session = requests.Session()
    my_headers = {
        "User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 14685.0.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.4992.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9"
    }
    extracted_content = []
    for url in URLs:
        try:
            if url.endswith('.pdf'):
                extracted_content.append(scrape_pdf(url))
                continue
            result = session.get(url, headers=my_headers, verify=False, timeout=3)
            doc = bs(result.content, "html.parser")
            contents = doc.find_all("p")
            for content in contents:
                extracted_content.append(content.text)
        except Exception as exc:
            print(f"Error extracting links from {url}: {exc}")
    return extracted_content

def scrape_pdf(url):
    session = requests.Session()
    my_headers = {
        "User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 14685.0.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.4992.0 Safari/537.36",
        "Accept": "application/pdf"
    }
    result = session.get(url, headers=my_headers, verify=False, timeout=3)
    with open("./temp.pdf", "wb") as f:
        f.write(result.content)
    extracted_texts = []
    with open("./temp.pdf", "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            extracted_texts.append(page.extract_text())
    extracted_text = " ".join(extracted_texts)
    return extracted_text

def extract_unique_urls(input_string):
    regex_patterns = [
        r'(https?://\S+)',
        r'(www\.\S+\.\w+)',
        r'(ftp://\S+)'
    ]

    extracted_urls = []
    for pattern in regex_patterns:
        matches = re.findall(pattern, input_string)
        extracted_urls.extend(matches)

    unique_urls = set(extracted_urls)
    return unique_urls

def web_search(query, relevanceSort=False):
    customer_message = query
    unique_urls = extract_unique_urls(customer_message)
    print(unique_urls)
    extracted_content = []
    if unique_urls:
        extracted_content = extract_links(list(unique_urls))
        print("Extracting content from URLs ")
    bi_encoder_searched_passages=""
    urls = []
    passages = []
    con = []
    start = time.time()
    search_results = list(search(customer_message, tld="com", num=10, stop=10, pause=0.75))
    for j in search_results:
        urls.append(j)
    print("URLS=", urls)
    DataWrtUrls = {}
    passages = []
    time_for_scraping = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(scraper, url, con, DataWrtUrls, passages): url for url in urls}
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f'URL {url} generated an exception: {exc}')
    
    #print("Passages=",passages)

    print("time for scraping: ",time.time()-time_for_scraping)
    passages2 = []
    i = 0
    try:
        while i < len(passages):
            P = ""
            while len(P.split()) <= 80 and i < len(passages):
                P += (passages[i] + " ")
                i += 1
            passages2.append(P.strip())
    except Exception as exc:
        print(f"Error processing passages: {exc}")

    if len(extracted_content) > 0:
        i = min(len(passages2), len(extracted_content))
        for j in range(i):
            passages2[j] = extracted_content[j]
            extracted_content.pop(j)
        for j in range(len(extracted_content)):
            passages2.append(extracted_content[j])


    if relevanceSort:
        bi_encoder_searched_passages = BM25func(passages2, customer_message)
    else:
        bi_encoder_searched_passages = passages2
    
    # if not prod: print(bi_encoder_searched_passages)

    end = time.time()

    # print(f"Runtime of the program is {end - start}")

    lfqa_time = time.time()

    question = customer_message
    print("Length of bi_encoder:", len(bi_encoder_searched_passages))
    if len(bi_encoder_searched_passages) >= 7:
        supporting_texts = "Supporting Text 1: " + str(bi_encoder_searched_passages[0]) + "\n" + \
                          "Supporting Text 2: " + str(bi_encoder_searched_passages[1]) + "\n" + \
                          "Supporting Text 3: " + str(bi_encoder_searched_passages[2]) + "\n" + \
                          "Supporting Text 4: " + str(bi_encoder_searched_passages[3]) + "\n" + \
                          "Supporting Text 5: " + str(bi_encoder_searched_passages[4]) + "\n" + \
                          "Supporting Text 6: " + str(bi_encoder_searched_passages[5]) + "\n" + \
                          "Supporting Text 7: " + str(bi_encoder_searched_passages[6])
    else:
        supporting_texts = ""
        for i in range(len(bi_encoder_searched_passages)):
            supporting_texts += "Supporting Text " + str(i + 1) + ": " + str(bi_encoder_searched_passages[i]) + "\n"

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful Research Assistant. Your job is to provide your boss with the most relevant information in a report format. If present, format code snippets within ''' ''' triple quotes."},
            {"role": "user", "content": "Generate answer to the question: " + str(question) + "\n\nSupporting Texts\n" + str(supporting_texts)}
        ],
        max_tokens=None,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|im_end|>"],
        stream=True
    )
    output = ""
    for token in response:
        if hasattr(token, 'choices'):
            output += token.choices[0].delta.content
            print(token.choices[0].delta.content, end='', flush=True)

    return output


if __name__ == "__main__":
    query = input("Enter your query: ")
    print(web_search(query))
