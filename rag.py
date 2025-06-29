import pickle
import chromadb
import time
import os

from pathlib import Path
from groq import Groq
from chromadbx import UUIDGenerator
from dotenv import load_dotenv
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, TesseractCliOcrOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem


load_dotenv()

IMAGE_RESOLUTION_SCALE = 2.0


GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def parse_docs(source):

    pipeline_options = PdfPipelineOptions()
    #pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    #pipeline_options.generate_page_images = True
    #pipeline_options.generate_picture_images = True
    #pipeline_options.do_ocr = True
    # ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True, tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    # ocr_options = EasyOcrOptions(force_full_page_ocr=True)
    # pipeline_options.ocr_options = ocr_options

    doc_converter = DocumentConverter(format_options = {
    InputFormat.PDF : PdfFormatOption(pipeline_options=pipeline_options)
    }
    )

    # Convert document
    doc = doc_converter.convert(source)
    doc_filename = doc.input.file.stem

    # # Save images of entire page created using ocr
    # for page_no, page in doc.document.pages.items():
    #     page_image_filename = Path(f"{doc_filename}-{page_no}.png")
    #     with page_image_filename.open("wb") as fp:
    #         page.image.pil_image.save(fp, format="PNG")

    # # Save images within document
    # picture_counter = 0
    # for element, _level in doc.document.iterate_items():
    #     if isinstance(element, PictureItem):
    #         picture_path = Path(f"{doc_filename}-picture-{picture_counter}.png")
    #         with picture_path.open("wb") as fp:
    #             element.get_image(doc.document).save(fp, "PNG")
    #         picture_counter += 1

    # images = []
    # for picture in doc.document.pictures:
    #     ref = picture.get_ref().cref
    #     image = picture.image
    #     if image:
    #         images.append(str(image.uri))

    # print(images[0])
    return doc.document



def summarize_images(images):

    client = Groq(api_key=GROQ_API_KEY)

    image_summarization_prompt = """
    Describe the image concisely in 1 to 2 sentences
    """

    image_summaries = []

    image = images[0]

    # messages = [
    #     {"role":"user", "content":image},
    #     {"role":"system", "content":image_summarization_prompt},
    # ]

    for image in images:
        messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": image_summarization_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image
                            }
                        }
                    ]
                }
            ]

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.5
        )
        summary = response.choices[0].message.content

        image_summaries.append(summary)



# with open("images.pkl","wb") as f:
#     pickle.dump(images, f)
# with open("images.pkl", "rb") as f:
#     images = pickle.load(f)
# print(len(images))
# summaries = summarize_images(images)
# print(summaries)

def replace_occurences(text, replacements):
    result = text

    IMAGE_PLACEHOLDER = "<!-- image_placeholder -->"

    for replacement in replacements:
        if IMAGE_PLACEHOLDER in result:
            result = result.replace(IMAGE_PLACEHOLDER, replacement, 1)
        else:
            break

    return result



def chunk_document(text):
    SPLIT_PATTERN = "\n#"
    chunks = text.split(SPLIT_PATTERN)
    print("Number of chunks generates is: ", len(chunks))

    return chunks


def _create_collection(client, collection_name):
    collection = client.get_or_create_collection(name=collection_name)
    return collection


def create_vectorstore(chunks, collection_name="test"):
    ids = UUIDGenerator(len(chunks))
    client = chromadb.PersistentClient()
    collection = _create_collection(client, collection_name=collection_name)
    collection.add(documents=chunks, ids=ids)

    return collection


def retrieve_relevant_chunks(query, collection):
    result = collection.query(
        query_texts=[query],
        n_results=1,
    ).get('documents')[0]

    return result


def generate_response(query, relevant_chunks):
    
    user_prompt = f"""
    Answer the question using following context:
    {relevant_chunks}
    - -
    Answer the question based on the above context: 
    {query}
    """

    messages = [
    {"role":"user", "content": user_prompt}
    ]
    
    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0,
    )

    return response.choices[0].message.content




if __name__ == "__main__":
    start = time.time()
    
    text = parse_docs()
    text = text.export_to_markdown()
    print(text)
    
    end = time.time()
    print("Time taken ", (end-start))
    
    chunks = chunk_document(text)
    collection = create_vectorstore(chunks, collection_name="penguins")

    query = "Can penguins fly?"
    
    relevant_chunks = retrieve_relevant_chunks(query=query, collection=collection)

    response = generate_response(query, relevant_chunks)

    print(response)






