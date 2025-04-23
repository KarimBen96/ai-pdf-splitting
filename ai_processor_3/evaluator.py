from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any
import sys
import os
import json
import re
from mistralai import Mistral
import datauri
from pprint import pprint

# from prompt_list import prompt_technical_sheets, prompt_table_of_contents, prompt_technical_sheets_mistral
# from document_processor import main

# from .prompt_list import (
#     prompt_technical_sheets,
#     prompt_table_of_contents,
#     prompt_technical_sheets_mistral,
# )
# from .document_processor import main


class Evaluator:
    """
    Integrates Mistral LLM for advanced boundary detection and content analysis
    of technical sheets in catalogs.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "mistral-small-latest",
        # model: str = "mistral-large-latest",
        # model: str = "pixtral-large-latest",
        prompt: str = "",
        debug: bool = False,
        detection_results: List = None,
        catalog_data: Dict = None,
    ):
        """
        Initialize the OCR analyzer.

        Args:
            api_key: Mistral API key (if None, uses environment variable)
            model: Model name to use (default: "mistral-small-latest")
            prompt: System prompt for the analysis
            debug: Enable debug output
        """
        self.debug = debug
        self.model = model
        self.system_prompt = prompt

        # Use provided API key or get from environment
        if api_key is None:
            api_key = os.environ.get("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError(
                    "API key must be provided or set as MISTRAL_API_KEY environment variable"
                )

        # Initialize Mistral client
        # self.llm = ChatMistralAI(model=model, temperature=0.2)
        self.client = Mistral(api_key=api_key)

        self.detection_results = detection_results
        self.catalog_data = catalog_data
        if catalog_data:
            self.total_pages = catalog_data["total_pages"]
            self.pages = catalog_data["pages"]


    def evaluate_with_llm(self, evaluation_report):
        """Send evaluation data to Mistral LLM for analysis."""

        # Uncomment and use the client initialization in __init__
        # self.client = Mistral(api_key=api_key)

        # Format the evaluation data for the LLM
        evaluation_data = json.dumps(evaluation_report, indent=2)

        # Create system prompt
        prompt = """
            You are an expert document evaluator specializing in technical catalogs and specification sheets.
            Your task is to evaluate the output of the document analyzer and provide feedback on its accuracy and completeness.
        
            Output format
            Provide your response in the following JSON format:
            {
                "overall_assessment": {
                    "quality": "Excellent/Good/Fair/Poor",
                    "detection_rate": "X/Y products correctly identified"
                },
                "product_feedback": [
                    {
                        "product": "product name",
                        "assessment": "evaluation of detection quality",
                        "confidence_level": "High/Medium/Low"
                    },
                ],
                "recommendations": "very breif suggestion for improvement",
            }
        """

        # Send to LLM
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Evaluate these technical sheet detection results:\n{evaluation_data}",
            },
        ]

        response = self.client.chat.complete(model=self.model, messages=messages)

        return response.choices[0].message.content

    
    def evaluate(self):
        evaluation_report = []

        for result in self.detection_results:
            product = result["product"]
            confidence = result["confidence"]
            pages = result["pages"]
            reason = result["reason"]

            # Check if the detected pages exist in the catalog
            valid_pages = [page for page in pages if page <= self.total_pages]
            if len(valid_pages) != len(pages):
                evaluation_report.append(
                    {
                        "product": product,
                        "status": "Invalid",
                        "reason": f"Some pages are out of range: {pages}",
                    }
                )
                continue

            # Check if "likely_technical_sheet" in the pages
            likely_technical_sheets = [
                self.pages[page_number - 1]["likely_technical_sheet"]
                for page_number in valid_pages
            ]
            if not any(likely_technical_sheets):
                evaluation_report.append(
                    {
                        "product": product,
                        "status": "Invalid",
                        "reason": f"No pages marked as likely technical sheet: {pages}",
                    }
                )
                continue

            # Check confidence
            if confidence < 0.6:
                evaluation_report.append(
                    {
                        "product": product,
                        "status": "Low Confidence",
                        "reason": f"Confidence level is low: {confidence}",
                    }
                )
                continue

            # Si tout est correct
            evaluation_report.append(
                {"product": product, "status": "Valid", "reason": reason}
            )

        # Generate AI-style evaluation
        llm_evaluation = self.evaluate_with_llm(evaluation_report)

        return evaluation_report, llm_evaluation
        return {"validation_results": evaluation_report, "ai_evaluation": llm_evaluation}


if __name__ == "__main__":
    detection_results = [
        {
            "product": "Barrière de sécurité T18",
            "confidence": 0.95,
            "pages": [5, 6],
            "reason": "Détection basée sur les caractéristiques techniques et les certificats.",
        },
        {
            "product": "Barrière de sécurité bois-métal TR18",
            "confidence": 0.85,
            "pages": [7, 8],
            "reason": "Détection basée sur les modèles et les conditions d'essais.",
        },
        {
            "product": "Barrière de sécurité bois-métal T22",
            "confidence": 0.85,
            "pages": [9, 10],
            "reason": "Détection basée sur les modèles et les conditions d'essais.",
        },
    ]

    catalog_data = {
        "total_pages": 10,
        "pages": [
            {
                "page_number": 1,
                "width": 595.2760009765625,
                "height": 841.8900146484375,
                "text_content": "S\u00c9CURIT\u00c9 ROUTI\u00c8RE\nGamme s\u00e9curit\u00e9 routi\u00e8re\nRoutes\nLarge gamme de barri\u00e8res de s\u00e9curit\u00e9 bois-m\u00e9tal  \nBarri\u00e8res m\u00e9talliques pour Ouvrages d\u2019Art  \nAtt\u00e9nuateurs de chocs redirectifs et bidirectionnels  \nPAGE-ID: 1\n<page:1>\n",
                "header_text": "S\u00c9CURIT\u00c9 ROUTI\u00c8RE Gamme   s\u00e9curit\u00e9 routi\u00e8re PAGE-ID: 1",
                "footer_text": "<page:1>",
                "has_tables": "true",
                "image_count": 3,
                "likely_technical_sheet": "false",
            },
            {
                "page_number": 2,
                "width": 595.2760009765625,
                "height": 841.8900146484375,
                "text_content": "Gamme s\u00e9curit\u00e9 routi\u00e8re\n \u2009\nX Barri\u00e8res de s\u00e9curit\u00e9 bois-m\u00e9tal\nMod\u00e8les  \nEntraxe \nsupport\nNiveau \nde retenue\nClasse de \nlargeur  de \nfonctionnement\n (W)\nNiveau de \ns\u00e9v\u00e9rit\u00e9\n(ASI)\nD\u00e9flexion \ndynamique\n(D)\nIntrusion \nv\u00e9hicule\n(VIn)\nNum\u00e9ro de \ncertificat CE\nT32\n1.33 m\nN2\nW2 (0.7 m)\nB\n0.6 m\nN.A\nn\u00b0 0497 CPR 5906\nT32\n2 m\nN2\nW2 (0.8 m)\nB\n0.7 m\nN.A\nn\u00b0 0497 CPR 5906\nT32\n2.66 m\nN2\nW3 (0.9 m)\nB\n0.8 m\nN.A\nn\u00b0 0497 CPR 5906\nTM18\n2 m\nN2\nW4 (1.3 m)\nA\n1.1 m\nVI3 (1.0 m)\nn\u00b0 1826 CPR 09 02 06 DR8\nT22\n2 m\nN2\nW4 (1.3 m)\nA\n1.0 m\nVI3 (0.9 m)\nn\u00b0 1826 CPR 09 02 06 DR7\nT22 BP\n2 m\nN2\nW4 (1.2 m)\nA\n1.0 m\nVI3 (0.9 m)\nn\u00b0 0497 CPR 7496\nT18\n2 m\nN2\nW5 (1.7 m) \nA\n1.5 m\nVI5 (1.7 m)\nn\u00b0 1826 CPR 09 02 06 DR3\nTR18\n2 m\nN2\nW5 (1.7 m)\nA\n1.5 m\nVI5 (1.7 m)\nn\u00b0 1826 CPR 09 02 06 DR11\nTM18\n4 m\nN2\nW5 (1.7 m)\nA\n1.2 m\nVI4 (1.3 m)\nn\u00b0 1826 CPR 09 02 06 DR06\nT22\n4 m\nN2\nW6 (1.9 m)\nA\n1.8 m\nVI6 (1.9 m)\nn\u00b0 1826 CPR 09 02 06 DR5\nT18\n4 m\nN2\nW7 (2.5 m)\nA\n2.3 m\nVI7 (2.3 m)\nn\u00b0 1826 CPR 09 02 06 DR4\nTR18\n4 m\nN2\nW7 (2.5 m)\nA\n2.3 m\nVI7 (2.3 m)\nn\u00b0 1826 CPR 09 02 06 DR12\nTM40\n2 m\nH2\nW5 (1.6 m)\nA\n1.5 m\nVI5 (1.7 m)\nn\u00b0 1826 CPR 09 02 06 DR10\nT40 BP\n2 m\nH2\nW5 (1.5 m)\nB\n1.3 m\nVI5 (1.7 m)\nn\u00b0 1826 CPR 09 02 06 DR2\nT40 \n2 m\nL2 =\nN2 + H2\nW3 (1.0 m) - N2\nW4 (1.2 m) - H2\nB\n0.5 m (N2)\n0.8 m (H2)\nVI3 (1.0 m)\nn\u00b0 1826 CPR 09 02 06 DR1\nPAGE-ID: 2\n<page:2>\n",
                "header_text": "Gamme s\u00e9curit\u00e9 routi\u00e8re  \u2009 X  Barri\u00e8res de s\u00e9curit\u00e9 bois-m\u00e9tal PAGE-ID: 2",
                "footer_text": "<page:2>",
                "has_tables": "true",
                "image_count": 2,
                "likely_technical_sheet": "true",
            },
            {
                "page_number": 3,
                "width": 595.2760009765625,
                "height": 841.8900146484375,
                "text_content": " \u2009\nX Barri\u00e8res de s\u00e9curit\u00e9 m\u00e9talliques Marcegaglia\nMod\u00e8les  \nOndes\nEntraxe \nsupport\nNiveau \nde retenue\nClasse de \nlargeur  de \nfonctionnement\n (W)\nNiveau de \ns\u00e9v\u00e9rit\u00e9\n(ASI)\nD\u00e9flexion \ndynamique\n(D)\nIntrusion \nv\u00e9hicule\n(VIn)\nNum\u00e9ro de \ncertificat CE\nMB-2601 (BL)\n2 n\n2 m\nN2\nW2 (0.8 m)\nA\n0.7 m\nN.A\nn\u00b0 408 2131 CPR 2014\nMB-2052 (BL)\n2 n\n4 m\nN2\nW4 (1.3 m)\nA\n1.1 m\nN.A\nn\u00b0282 2131 CPR 2014\nMB-446 (BL)\n3 n\n1.5 m\nH2\nW4 (1.3 m)\nA\n1.1 m\nVI4 (1.2 m)\nn\u00b0226 2131 CPR 2015\nMB-1292 (BP)\n3 n\n2.25 m\nH2\nW4 (1.3 m)\nA\n1 m\nVI4 (1.2 m)\nn\u00b0169 2131 CPR 2014\nMB-2052 (BL)\n3 n\n2.25 m\nH3\nW5 (1.6 m)\nA\n1.2 m\nVI5 (1.7 m)\nn\u00b0362 2131 CPR 2014\nMB-1622 (BP)\n3 n\n2.25 m\nH3\nW5 (1.7 m) \nB\n1.4 m\nVI6 (1.8 m)\nn\u00b0215 2131 CPR 2014\nMB-2417 (BL)\n3 n\n1.5 m\nH4\nW5 (1.7 m)\nA\n1.4 m\nVI6 (2.1 m)\nn\u00b0388 2131 CPR 2013\nMB-1631 (BP)\n3 n\n1.5 m\nH4\nW4 (1.3 m)\nB\n1.0 m\nVI7 (2.2 m)\nn\u00b0216 2131 CPR 2014\n \u2009\nX Att\u00e9nuateurs de chocs EURO-TRACC\u2122\n Mod\u00e8les\nNiveau de \nperformance\nNiveau de s\u00e9v\u00e9rit\u00e9 de \nchoc (ASI)\nClasse de zone \nde r\u00e9orientation\nClasse de zone \nde d\u00e9placement lat\u00e9ral\nNum\u00e9ro de \ncertificat CE\nEURO-TRACC\u2122\n110 droit\n110\nA\nZ1\nD1\nn\u00b00402 CPR 495804\nEURO-TRACC\u2122\n110 wide\n110\nB\nZ1\nD1\nn\u00b00402 CPR 495804\nEURO-TRACC\u2122\n100 droit\n100\nB\nZ1\nD1\nn\u00b00402 CPR 495804\nEURO-TRACC\u2122\n100 wide\n100\nB\nZ1\nD1\nn\u00b00402 CPR 495804\nEURO-TRACC\u2122\n80 droit\n80\nA\nZ1\nD1\nn\u00b00402 CPR 495804\nEURO-TRACC\u2122\n80 wide\n80\nB\nZ1\nD1\nn\u00b00402 CPR 495804\nPAGE-ID: 3\n<page:3>\n",
                "header_text": "PAGE-ID: 3",
                "footer_text": "MB-2417 (BL) 3 n 1.5 m H4 W5 (1.7 m) A 1.4 m VI6 (2.1 m) n\u00b0388 2131 CPR 2013 MB-1631 (BP) 3 n 1.5 m H4 W4 (1.3 m) B 1.0 m VI7 (2.2 m) n\u00b0216 2131 CPR 2014 <page:3>",
                "has_tables": "true",
                "image_count": 2,
                "likely_technical_sheet": "true",
            },
            {
                "page_number": 4,
                "width": 595.2760009765625,
                "height": 841.8900146484375,
                "text_content": "1 route de Tertu - 61160 Villedieu-L\u00e8s-Bailleul - FRANCE\nT\u00e9l. : 02 33 36 11 02 - tertu.com\nGamme s\u00e9curit\u00e9 routi\u00e8re\n \u2009\nX R\u00e9glementation Nationale des Equipements Routiers\nConfiguration \nen accotement\nNiveau de \nretenue mini\nLimitation de vitesse  \n\u2265 70 km/h et < 80 km/h ou  \nV851 < 80 km/h\nN1\nLimitation de vitesse \u2265 80 km/h \nN2\nAccotement r\u00e9duit\nN2\nObstacle saillant\nN2\nD\u00e9nivel\u00e9 important \nH2\nDanger pour autres \nutilisateurs ou riverains\nH2\nConfiguration en Terre \nPlein Central\nNiveau de retenue \nminimun\nLimitation de  \nvitesse \u2265 70 km/h et  \n< 90 km/h\nN2\nLimitation de vitesse  \n\u2265 90 km/h et  \nLargeur TPC < 5m\nChauss\u00e9es \ns\u00e9par\u00e9es\n2*1 ou \n2+1 \n2*2\n2*3 \net +\nN2\nN2\nH1\nH2\nLargeur TPC \u2265 5m\nN2\nBARRI\u00c8RES DE S\u00c9CURIT\u00c9 Tableau r\u00e9sum\u00e9 de l\u2019arr\u00eat\u00e9 du 04/07/2019\nATTENUATEURS DE CHOCS \nSections limit\u00e9es \u00e0 : \nNiveau mini \nde retenue\n70 km/h \n80/1\n90 km/h \n80\n110 km/h\n100\n130 km/h\n110\nORIGINES DE FILE PERFORMANTES \nSections limit\u00e9es \u00e0 : \nNiveau mini\n de retenue\n70 km/h \nP1\n90 km/h \nP2\n110 km/h\nP3\n130 km/h\nP4\nRACCORDEMENTS NF 058\nFamille \nde produits\nNiveau \nde retenue\n\u2206 D\nPi\u00e8ce de \nraccordement \nsp\u00e9cifique\nClasses d\u2019\u00e9valuation\nIdentique\nIdentique\n\u2264 50 cm\nNon\nPas d\u2019\u00e9valuation particuli\u00e8re\nA\n\u2264 50 cm\nNon\nSimulations num\u00e9riques\nB\nIdentique\nDiff\u00e9rent\n (sauf niveau L)\nX\nOui/Non\nSimulations num\u00e9riques\nB\nDiff\u00e9rente \nIdentique\n\u2264 50 cm\nNon\nPas d\u2019\u00e9valuation particuli\u00e8re\nA\nOui\nSimulations num\u00e9riques\nB\n> 50 cm\nNon\nSimulations num\u00e9riques\nB\nOui \nEssai de choc physique + simulations \nnum\u00e9riques\nC\nDiff\u00e9rente\nDiff\u00e9rent\n(sauf niveau L)\nX\nOui/Non\nEssai de choc physique + simulations \nnum\u00e9riques\nC\n1 V85 : vitesse th\u00e9orique en dessous de laquelle circule 85% des usagers\nCr\u00e9dits photos et conception : Tertu SAS \nPAGE-ID: 4\n<page:4>\n",
                "header_text": "Gamme s\u00e9curit\u00e9 routi\u00e8re  \u2009 X  R\u00e9glementation Nationale des Equipements Routiers PAGE-ID: 4",
                "footer_text": "1 route de Tertu - 61160 Villedieu-L\u00e8s-Bailleul - FRANCE T\u00e9l. : 02 33 36 11 02 -  tertu.com Diff\u00e9rente Diff\u00e9rent (sauf niveau L) X Oui/Non Essai de choc physique + simulations  num\u00e9riques C Cr\u00e9dits photos et conception : Tertu SAS  <page:4>",
                "has_tables": "true",
                "image_count": 4,
                "likely_technical_sheet": "true",
            },
            {
                "page_number": 5,
                "width": 595.2760009765625,
                "height": 841.8900146484375,
                "text_content": "S\u00c9CURIT\u00c9 ROUTI\u00c8RE\nBarri\u00e8re de s\u00e9curit\u00e9 bois-m\u00e9tal T18\nN2 W7, support tous les 4 m \nN2 W5, support tous les 2 m \nASI A\nSection courante\nT18\nPAGE-ID: 5\n<page:5>\n",
                "header_text": "S\u00c9CURIT\u00c9 ROUTI\u00c8RE Barri\u00e8re de s\u00e9curit\u00e9 bois-m\u00e9tal   T18 PAGE-ID: 5",
                "footer_text": "<page:5>",
                "has_tables": "true",
                "image_count": 4,
                "likely_technical_sheet": "true",
            },
            {
                "page_number": 6,
                "width": 595.2760009765625,
                "height": 841.8900146484375,
                "text_content": "Barri\u00e8re de s\u00e9curit\u00e9 bois-m\u00e9tal T18\n \u2009\nX Conditions d\u2019essais\nNiveau de \nretenue\nCrit\u00e8res relatifs aux essais de choc\nEnergie \ncin\u00e9tique\nNiveau \nd\u2019utilisation\n recommand\u00e9\nEssai\nVitesse \nd\u2019impact\nAngle \nd\u2019impact\nMasse \ntotale \nv\u00e9hicule\nN2\nTB 32\n110 km/h\n20\u00b0\n1500 kg\n81,9 kJ\nRetenue normale\nN2\nTB 11\n100 km/h\n20\u00b0\n900 kg\n40,6 kJ\nRetenue normale\n \u2009\nX Caract\u00e9ristiques techniques\n\u2022\t\u00c9l\u00e9ments de glissement constitu\u00e9s d\u2019un rondin en bois de 18 cm de diam\u00e8tre \net de 4\u00a0m de long, renforc\u00e9 par un fer en U inclus dans le bois en face arri\u00e8re\n\u2022\tSupports m\u00e9talliques C100 espac\u00e9s tous les 4 m, ou tous les 2 m, avec \u00e9carteur \nen bois de 18 cm de diam\u00e8tre  \n\u2022\tBois trait\u00e9 en autoclave sans chrome, ni arsenic\n \u2009\nX Principe de montage\n \u2009\nX Certificats\n\u2022\tToutes nos barri\u00e8res de s\u00e9curit\u00e9 font l\u2019objet d\u2019une certification CE en conformit\u00e9 avec la norme EN 1317 1-2-5\n\u2022\tT18 4m : CE n\u00b0 1826 CPR 09 02 06 DR4\n\u2022\tT18 4ms2 : CE n\u00b0 1826 CPR 09 02 06 DR3\n\u2022\tOrganisme certificateur : ASCQUER\n1 route de Tertu - 61160 Villedieu-L\u00e8s-Bailleul - FRANCE\nT\u00e9l. : 02 33 36 11 02 - tertu.com\n720 mm\n15 mm\n15 mm\n700 mm (+0 et -50)\nMod\u00e8le\nLargeur de \nfonctionnement\n(W)\nD\u00e9flexion \ndynamique \n(D)\nNiveau \nde s\u00e9v\u00e9rit\u00e9\n(ASI)\nIntrusion \nv\u00e9hicule\n(Vin)\nLongueur minimale \nrecommand\u00e9e et \nextr\u00e9mit\u00e9s (ml)\nT18 4m\nW7 (2.5 m)\n2.3 m\nA (\u2264 1.0)\nVI7 (2.3 m)\n60 ml + (2*4 ml)\nT18 4ms2\nW5 (1.7 m)\n1.5 m\nA (\u2264 1.0)\nVI5 (1.7 m)\n60 ml + (2*4 ml)\nCr\u00e9dits photos et conception : Tertu SAS\nPAGE-ID: 6\n<page:6>\n",
                "header_text": "Barri\u00e8re de s\u00e9curit\u00e9 bois-m\u00e9tal T18  \u2009 X  Conditions d\u2019essais PAGE-ID: 6",
                "footer_text": "1 route de Tertu - 61160 Villedieu-L\u00e8s-Bailleul - FRANCE T\u00e9l. : 02 33 36 11 02 -  tertu.com Cr\u00e9dits photos et conception : Tertu SAS <page:6>",
                "has_tables": "true",
                "image_count": 10,
                "likely_technical_sheet": "true",
            },
            {
                "page_number": 7,
                "width": 595.2760009765625,
                "height": 841.8900146484375,
                "text_content": "S\u00c9CURIT\u00c9 ROUTI\u00c8RE\nBarri\u00e8re de s\u00e9curit\u00e9 bois-m\u00e9tal TR18\nN2 W7, support tous les 4 m \nN2 W5, support tous les 2 m \nASI A\nSection courante\nTR18\nPAGE-ID: 7\n<page:7>\n",
                "header_text": "S\u00c9CURIT\u00c9 ROUTI\u00c8RE Barri\u00e8re de s\u00e9curit\u00e9 bois-m\u00e9tal  TR18 PAGE-ID: 7",
                "footer_text": "<page:7>",
                "has_tables": "true",
                "image_count": 4,
                "likely_technical_sheet": "true",
            },
            {
                "page_number": 8,
                "width": 595.2760009765625,
                "height": 841.8900146484375,
                "text_content": "Barri\u00e8re de s\u00e9curit\u00e9 bois-m\u00e9tal TR18\n \u2009\nX Conditions d\u2019essais\nNiveau de \nretenue\nCrit\u00e8res relatifs aux essais de choc\nEnergie \ncin\u00e9tique\nNiveau \nd\u2019utilisation\n recommand\u00e9\nEssai\nVitesse \nd\u2019impact\nAngle \nd\u2019impact\nMasse \ntotale \nv\u00e9hicule\nN2\nTB 32\n110 km/h\n20\u00b0\n1500 kg\n81,9 kJ\nRetenue normale\nN2\nTB 11\n100 km/h\n20\u00b0\n900 kg\n40,6 kJ\nRetenue normale\nMod\u00e8le\nLargeur de \nfonctionnement\n(W)\nD\u00e9flexion \ndynamique \n(D)\nNiveau \nde s\u00e9v\u00e9rit\u00e9\n(ASI)\nIntrusion \nv\u00e9hicule\n(Vin)\nLongueur minimale \nrecommand\u00e9e et \nextr\u00e9mit\u00e9s (ml)\nTR18 4m\nW7 (2.5 m)\n2.3 m\nA (\u2264 1.0)\nVI7 (2.3 m)\n60 ml + (2*4 ml)\nTR18 4ms2\nW5 (1.7 m)\n1.5 m\nA (\u2264 1.0)\nVI5 (1.7 m)\n60 ml + (2*4 ml)\n \u2009\nX Certificats\n\u2022\tToutes nos barri\u00e8res de s\u00e9curit\u00e9 font l\u2019objet d\u2019une certification CE en conformit\u00e9 avec la norme EN 1317 1-2-5\n\u2022\tTR18 4m : CE n\u00b0 1826 CPR 09 02 06 DR12\n\u2022\tTR18 4ms2 : CE n\u00b0 1826 CPR 09 02 06 DR11\n\u2022\tOrganisme certificateur : ASCQUER\n1 route de Tertu - 61160 Villedieu-L\u00e8s-Bailleul - FRANCE\nT\u00e9l. : 02 33 36 11 02 - tertu.com\n \u2009\nX Caract\u00e9ristiques techniques\n\u2022\t\u00c9l\u00e9ments de glissement constitu\u00e9s d\u2019une section en bois rabot\u00e9 de 170*145\u00a0mm, de \n4 m de long (2 m pour le mod\u00e8le 4ms2), renforc\u00e9s par un fer en U de 4\u00a0m inclus dans \nle bois \u00e0 la face arri\u00e8re\n\u2022\tSupports m\u00e9talliques C100 espac\u00e9s tous les 4 m, ou tous les 2 m, habill\u00e9s avec \n\u00e9carteurs en bois, section 170 x 180 x 730 \n\u2022\tBois trait\u00e9 en autoclave sans chrome ni arsenic\n700 mm\n \u2009\nX Principe de montage\n720 mm\n15 mm\n15 mm\n330 mm\nCr\u00e9dits photos et conception : Tertu SAS \nPAGE-ID: 8\n<page:8>\n",
                "header_text": "Barri\u00e8re de s\u00e9curit\u00e9 bois-m\u00e9tal TR18  \u2009 X  Conditions d\u2019essais PAGE-ID: 8",
                "footer_text": "1 route de Tertu - 61160 Villedieu-L\u00e8s-Bailleul - FRANCE T\u00e9l. : 02 33 36 11 02 -  tertu.com Cr\u00e9dits photos et conception : Tertu SAS  <page:8>",
                "has_tables": "true",
                "image_count": 8,
                "likely_technical_sheet": "true",
            },
            {
                "page_number": 9,
                "width": 595.2760009765625,
                "height": 841.8900146484375,
                "text_content": "S\u00c9CURIT\u00c9 ROUTI\u00c8RE\nBarri\u00e8re de s\u00e9curit\u00e9 bois-m\u00e9tal T22\nN2 W6, support tous les 4 m\nN2 W4, support tous les 2 m\nASI A\nSection courante\nT22\nPAGE-ID: 9\n<page:9>\n",
                "header_text": "S\u00c9CURIT\u00c9 ROUTI\u00c8RE Barri\u00e8re de s\u00e9curit\u00e9 bois-m\u00e9tal  T22 PAGE-ID: 9",
                "footer_text": "<page:9>",
                "has_tables": "true",
                "image_count": 4,
                "likely_technical_sheet": "true",
            },
            {
                "page_number": 10,
                "width": 595.2760009765625,
                "height": 841.8900146484375,
                "text_content": "Barri\u00e8re de s\u00e9curit\u00e9 bois-m\u00e9tal T22\n \u2009\nX Conditions d\u2019essais\nNiveau de \nretenue\nCrit\u00e8res relatifs aux essais de choc\nEnergie \ncin\u00e9tique\nNiveau \nd\u2019utilisation\n recommand\u00e9\nEssai\nVitesse \nd\u2019impact\nAngle \nd\u2019impact\nMasse \ntotale \nv\u00e9hicule\nN2\nTB 32\n110 km/h\n20\u00b0\n1500 kg\n81,9 kJ\nRetenue normale\nN2\nTB 11\n100 km/h\n20\u00b0\n900 kg\n40,6 kJ\nRetenue normale\nMod\u00e8le\nLargeur de \nfonctionnement\n(W)\nD\u00e9flexion \ndynamique \n(D)\nNiveau \nde s\u00e9v\u00e9rit\u00e9\n(ASI)\nIntrusion \nv\u00e9hicule\n(Vin)\nLongueur minimale \nrecommand\u00e9e et \nextr\u00e9mit\u00e9s (ml)\nT22 4m\nW6 (1.9 m)\n1.8 m\nA (\u2264 1.0)\nVI6 (1.9 m)\n72 ml + ( 2*4 ml)\nT22 4ms2\nW4 (1.3 m)\n1.0 m\nA (\u2264 1.0)\nVI3 (0.9 m)\n72 ml + ( 2*4 ml)\n \u2009\nX Caract\u00e9ristiques techniques\n\u2022\t\u00c9l\u00e9ments de glissement constitu\u00e9s d\u2019un demi-rondin de 22 cm de diam\u00e8tre \net de 4 m de long renforc\u00e9 par un fer en U inclus dans le bois en face arri\u00e8re\n\u2022\tSupports m\u00e9talliques C100 espac\u00e9s tous les 4 m, ou tous les 2 m, avec \n\u00e9carteur en bois de 18 cm de diam\u00e8tre  \n\u2022\tBois trait\u00e9 en autoclave sans chrome ni arsenic\n \u2009\nX Principe de montage\n \u2009\nX Certificats\n\u2022\tToutes nos barri\u00e8res de s\u00e9curit\u00e9 font l\u2019objet d\u2019une certification CE en conformit\u00e9 avec la norme  \nNF EN 1317 1-2-5\n\u2022\tT22 4m : CE n\u00b0 1826 CPR 09 02 06 DR5\n\u2022\tT22 4ms2 : CE n\u00b0 1826 CPR 09 02 06 DR7\n\u2022\tOrganisme certificateur : ASCQUER\n1 route de Tertu - 61160 Villedieu-L\u00e8s-Bailleul - FRANCE\nT\u00e9l. : 02 33 36 11 02 - tertu.com\n15 mm\n15 mm\n700 mm \n(+0 et -50)\n610 mm \n(\u00b1 15 mm)\nCr\u00e9dits photos et conception : Tertu SAS \nPAGE-ID: 10\n<page:10>\n",
                "header_text": "Barri\u00e8re de s\u00e9curit\u00e9 bois-m\u00e9tal T22  \u2009 X  Conditions d\u2019essais PAGE-ID: 10",
                "footer_text": "1 route de Tertu - 61160 Villedieu-L\u00e8s-Bailleul - FRANCE T\u00e9l. : 02 33 36 11 02 -  tertu.com Cr\u00e9dits photos et conception : Tertu SAS  <page:10>",
                "has_tables": "true",
                "image_count": 8,
                "likely_technical_sheet": "true",
            },
        ],
    }

    evaluator = Evaluator(
        detection_results=detection_results, catalog_data=catalog_data
    )
    evaluation_report, llm_evaluation = evaluator.evaluate()
    pprint(evaluation_report)
    print("\n\n\n")
    print(llm_evaluation)
