import base64
import json
import os
from datetime import date
from io import BytesIO
from typing import List, Union

import fitz
import instructor
import streamlit as st
from PIL import Image
from openai import OpenAI
from pydantic import BaseModel, Field


def custom_json_encoder(obj):
    """Custom JSON encoder for handling non-serializable types."""
    if isinstance(obj, date):
        return obj.isoformat()  # Convert date to ISO format
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def dicts_to_jsonl(data_list: List[dict], dir_path: str, filename: str) -> None:
    """Save a list of dictionaries to a JSONL file in a specified directory."""
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, filename)

    if not file_path.endswith('.jsonl'):
        file_path += '.jsonl'

    with open(file_path, 'w', encoding='utf-8') as out:
        for ddict in data_list:
            json_line = json.dumps(ddict, default=custom_json_encoder)  # Use custom encoder
            out.write(json_line + '\n')


def pdf_to_base64(pdf_path: str, page: int):
    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    # Get the first page
    page = pdf_document.load_page(page)  # 0 means the first page

    # Convert page to image
    pix = page.get_pixmap()

    # Convert pixmap to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Convert image to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Close the PDF
    pdf_document.close()

    return img_str


def extract_structured_info_from_pdf(pdf_path: str, data_type: str, openai_api_key: str,
                                     llm_model_name: str = "gpt-4o"):
    # Define the schemas for the structured response
    class Name(BaseModel):
        name: str = Field(description="Der Name, Titel oder die Bezeichnung des Leuchmittels")

    class ElectricalData(BaseModel):
        nennstrom: float = Field(description="Der Nennstrom des Lechtmittels in Ampere")
        min_stromsteuerbereich: int = Field(description="Das Minimum des Stromsteuerbereichs in Ampere")
        max_stromsteuerbereich: int = Field(description="Das Maximum des Stromsteuerbereichs in Ampere")
        nennleistung: float = Field(description="Die Nennleistung des Lechtmittels in Watt")
        nennspannung: float = Field(description="Die Nennspannung des Lechtmittels in Volt")

    class SizeAndWeightData(BaseModel):
        durchmesser: float = Field(description="Der Durchmesser des Lechtmittels in Millimeter")
        laenge: float = Field(description="Die Länge des Lechtmittels in Millimeter")
        laenge_sockel: float = Field(
            description="Die Länge des Lechtmittels mit Sockel jedoch ohne Sockelstift in Millimeter")
        abstand_lichtschwerpunkt: float = Field(
            description="Der Abstand Lichtschwerpunkt (LCL) des Lechtmittels in Millimeter")
        elektrodenabstand_kalt: float = Field(description="Der Elektrodenabstand kalt des Lechtmittels in Millimeter")
        produktgewicht: float = Field(description="Das Produktgewicht des Lechtmittels in Gramm")
        kabel_laenge: Union[float, None] = Field(
            description="Falls vorhanden, die Kabellänge des Lechtmittels. Ansonsten None.")

    class TemperatureData(BaseModel):
        max_temp: int = Field(
            description="Die maximal zulässige Umgebungstemperatur Quetschung des Lechtmittels in Grad Celsius")

    class LifetimeData(BaseModel):
        lifetime: int = Field(description="Die Lebensdauer des Lechtmittels in Stunden/h")
        warranty: int = Field(description="Die Service Warranty Lifetime des Lechtmittels in Stunden/h")

    class AdditionalProductData(BaseModel):
        sockel_anode: str = Field(description="Die Sockel Anode (Normbezeichnung) des Lechtmittels")
        sockel_kathode: str = Field(description="Die Sockel Kathode (Normbezeichnung) des Lechtmittels")
        anmerkung_produkt: str = Field(description="Die Anmerkung zum Produkt des Lechtmittels")

    class UsageData(BaseModel):
        kuehlung: str = Field(description="Die Kühlung des Lechtmittels")
        brennstellung: str = Field(description="Die Brennstellung des Lechtmittels")

    class EnvironmentData(BaseModel):
        datum_deklaration: date = Field(description="Das Datum der Deklaration des Lechtmittels im DD-MMM-YYYY Format")
        erzeugnisnummer: List[int] = Field(
            description="Die Primäre Erzeugnisnummer(n) des Lechtmittels. Können eine oder mehrere Nummern getrennt duch '|' sein.",
            min_items=1)
        stoff_kandidatenliste: str = Field(description="Der Stoff der Kandidatenliste 1 des Lechtmittels")
        stoff_cas_nr: str = Field(description="Die CAS Nr. des Stoffes 1 des Lechtmittels")
        info_sicherer_gebrauch: str = Field(description="Die nformationen zum sicheren Gebrauch des Lechtmittels")
        scip_deklarationsnummer: List[str] = Field(
            description="Die SCIP Deklarationsnummer(n) des Lechtmittels. Können eine oder mehrere Nummern getrennt duch '|' sein.",
            min_items=1)

    if data_type == 'name':
        response_model = Name
        page = 0

    elif data_type == 'electrical_data':
        response_model = ElectricalData
        page = 1

    elif data_type == 'size_and_weight':
        response_model = SizeAndWeightData
        page = 1

    elif data_type == 'temperature_data':
        response_model = TemperatureData
        page = 1

    elif data_type == 'lifetime_data':
        response_model = LifetimeData
        page = 1

    elif data_type == 'additional_product_data':
        response_model = AdditionalProductData
        page = 1

    elif data_type == 'usage_data':
        response_model = UsageData
        page = 2

    elif data_type == 'environment_data':
        response_model = EnvironmentData
        page = 2

    # Get base64 encoded image
    base64_image = pdf_to_base64(pdf_path=pdf_path, page=page)

    llm_client = instructor.from_openai(OpenAI(api_key=openai_api_key))

    prompt = [
        {
            "type": "text",
            "text": "Extrahiere die Eigenschaften und Daten des Leuchtmittels aus dem Bild des Produktdatenblattes."
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
    ]

    # query the OpenAI model and get a structured response
    response = llm_client.chat.completions.create(
        model=llm_model_name,
        messages=[{"role": "user", "content": prompt}],
        response_model=response_model,
        temperature=0
    )

    return response


def pdf_to_dict(pdf_path: str, openai_api_key: str, llm_model_name: str = "gpt-4o"):
    data = {}
    info_types = ["name", "electrical_data", "size_and_weight", "temperature_data", "lifetime_data",
                  "additional_product_data", "usage_data", "environment_data"]

    for info_type in info_types:
        extracted_data = extract_structured_info_from_pdf(pdf_path=pdf_path, data_type=info_type,
                                                          openai_api_key=openai_api_key, llm_model_name=llm_model_name)
        data.update(extracted_data.dict())

    return data


def pdfs_to_jsonl(pdfs_dir: str, jsonl_dir: str, jsonl_filename: str, openai_api_key: str,
                  llm_model_name: str = "gpt-4o"):
    data_dict_list = []
    for file in os.listdir(pdfs_dir):
        if file.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdfs_dir, file)
            file_name = pdf_path.split("/")[-1]
            data_dict = pdf_to_dict(pdf_path=pdf_path, openai_api_key=openai_api_key, llm_model_name=llm_model_name)
            data_dict['file_name'] = file_name
            data_dict_list.append(data_dict)

    dicts_to_jsonl(data_list=data_dict_list, dir_path=jsonl_dir, filename=jsonl_filename)


# extract information from multiple PDFs to a single JSONL file
pdfs_to_jsonl(pdfs_dir=os.getcwd() + "/data", jsonl_dir=os.getcwd() + "/data", jsonl_filename="illuminants.jsonl",
              openai_api_key=st.secrets["OPENAI_API_KEY"], llm_model_name="gpt-4o")
