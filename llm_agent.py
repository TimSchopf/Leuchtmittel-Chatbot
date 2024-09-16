from datetime import date
from typing import List, Dict, TypedDict, Literal, Union

import instructor
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from openai import OpenAI
from pydantic import BaseModel, Field

from filtering import filter_dataframe


class AgentState(TypedDict):
    messages: List[HumanMessage | AIMessage | SystemMessage]
    response: str
    chat_mode: str
    data: pd.DataFrame
    retrieved_data: List[dict]


class LLMAgent:
    def __init__(self, openai_api_key: str):
        self.openai_key = openai_api_key
        self.llm_model_name = "gpt-4o"
        self.llm_client = instructor.from_openai(OpenAI(api_key=self.openai_key))
        self.state: AgentState = self._initialize_state()
        self.workflow = self._create_workflow()

    def _initialize_state(self) -> AgentState:
        """Initialize the agent state."""
        return {
            "messages": [],
            "chat_mode": "chit-chat",
            "response": None,
            "data": pd.read_json("data/illuminants.jsonl", lines=True),
            "retrieved_data": None
        }

    def _create_workflow(self):
        workflow = StateGraph(AgentState)

        def select_chat_mode(state: AgentState) -> AgentState:
            class ChatMode(BaseModel):
                chat_mode: Literal[tuple(["chit-chat", "retrieval"])] = Field(
                    description="Der Modus in welchen der Chatbot gehen soll, basierend auf der Chat Historie.")

            prompt = f"""Analysiere die Chat Historie und Entscheide welche Chat Modus benötigt wird. Falls ein Nutzer Informationen über Leuchtmittel benötigt (zum Beispiel falls er fragt 'Wie viel wiegt XBO 4000 W/HS XL OFR?') müssen diese aus einer Datenbank retrieved werden. Schalte in diesem Fall in den 'retrieval' Modus. Falls keine neuen Informationen über Leuchtmittel aus der Datenbank benötigt werden (zum Beispiel weil es sich um eine eine chit-chat query handelt), schalte in den 'chit-chat' Modus. 

            Chat Historie: {state['messages']}"""

            # query the OpenAI model and get a structured response
            response = self.llm_client.chat.completions.create(
                model=self.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                response_model=ChatMode,
                temperature=0
            )

            state['chat_mode'] = response.chat_mode
            return state

        def retrieve(state: AgentState) -> AgentState:
            prompt = f"""Basierend auf der Nutzeranfrage, enzscheide welche Informationen benötigt werden und extrahiere diese. 

                        Nutzeranfrage: {state['messages']}"""

            class FloatSearchOperator(BaseModel):
                operator: Literal['==', '>', '>=', '<', '<='] = Field(...,
                                                                      description="Der Such bzw. Vergleichs Operator. Default zu exakt Match Operator.")
                value: float = Field(..., description="The numerische Wert.")

            class IntSearchOperator(BaseModel):
                operator: Literal['==', '>', '>=', '<', '<='] = Field(...,
                                                                      description="Der Such bzw. Vergleichs Operator. Default zu exakt Match Operator.")
                value: float = Field(..., description="The numerische Wert.")

            class DateSearchOperator(BaseModel):
                operator: Literal['==', '>', '>=', '<', '<='] = Field(...,
                                                                      description="Der Such bzw. Vergleichs Operator. Default zu exakt Match Operator.")
                value: date = Field(..., description="Das Datum im DD-MMM-YYYY Format")

            class ExtractData(BaseModel):
                name: Union[str, None] = Field(
                    description="Der Name, Titel oder die Bezeichnung des Leuchmittels. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                nennstrom: Union[FloatSearchOperator, None] = Field(
                    description="Der Nennstrom des Lechtmittels in Ampere. Dazu der Such Operator, ob ein exaktes Match (==), größer (>), größer gleich (>=), kleiner gleich (<=) oder kleiner (<) als dieser Wert gesucht wird. Zum Beispiel (10.0, >=). Default zu exakt Match Operator. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                min_stromsteuerbereich: Union[IntSearchOperator, None] = Field(
                    description="Das Minimum des Stromsteuerbereichs in Ampere. Dazu der Such Operator, ob ein exaktes Match (==), größer (>), größer gleich (>=), kleiner gleich (<=) oder kleiner (<) als dieser Wert gesucht wird. Zum Beispiel (10, >=). Default zu exakt Match Operator. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                max_stromsteuerbereich: Union[IntSearchOperator, None] = Field(
                    description="Das Maximum des Stromsteuerbereichs in Ampere. Dazu der Such Operator, ob ein exaktes Match (==), größer (>), größer gleich (>=), kleiner gleich (<=) oder kleiner (<) als dieser Wert gesucht wird. Zum Beispiel (10, >=). Default zu exakt Match Operator. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                nennleistung: Union[FloatSearchOperator, None] = Field(
                    description="Die Nennleistung des Lechtmittels in Watt. Dazu der Such Operator, ob ein exaktes Match (==), größer (>), größer gleich (>=), kleiner gleich (<=) oder kleiner (<) als dieser Wert gesucht wird. Zum Beispiel (10.0, >=). Default zu exakt Match Operator. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                nennspannung: Union[FloatSearchOperator, None] = Field(
                    description="Die Nennspannung des Lechtmittels in Volt. Dazu der Such Operator, ob ein exaktes Match (==), größer (>), größer gleich (>=), kleiner gleich (<=) oder kleiner (<) als dieser Wert gesucht wird. Zum Beispiel (10.0, >=). Default zu exakt Match Operator. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                durchmesser: Union[FloatSearchOperator, None] = Field(
                    description="Der Durchmesser des Lechtmittels in Millimeter. Dazu der Such Operator, ob ein exaktes Match (==), größer (>), größer gleich (>=), kleiner gleich (<=) oder kleiner (<) als dieser Wert gesucht wird. Zum Beispiel (10.0, >=). Default zu exakt Match Operator. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                laenge: Union[FloatSearchOperator, None] = Field(
                    description="Die Länge des Lechtmittels in Millimeter. Dazu der Such Operator, ob ein exaktes Match (==), größer (>), größer gleich (>=), kleiner gleich (<=) oder kleiner (<) als dieser Wert gesucht wird. Zum Beispiel (10.0, >=). Default zu exakt Match Operator. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                laenge_sockel: Union[FloatSearchOperator, None] = Field(
                    description="Die Länge des Lechtmittels mit Sockel jedoch ohne Sockelstift in Millimeter. Dazu der Such Operator, ob ein exaktes Match (==), größer (>), größer gleich (>=), kleiner gleich (<=) oder kleiner (<) als dieser Wert gesucht wird. Zum Beispiel (10.0, >=). Default zu exakt Match Operator. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                abstand_lichtschwerpunkt: Union[FloatSearchOperator, None] = Field(
                    description="Der Abstand Lichtschwerpunkt (LCL) des Lechtmittels in Millimeter. Dazu der Such Operator, ob ein exaktes Match (==), größer (>), größer gleich (>=), kleiner gleich (<=) oder kleiner (<) als dieser Wert gesucht wird. Zum Beispiel (10.0, >=). Default zu exakt Match Operator. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                elektrodenabstand_kalt: Union[FloatSearchOperator, None] = Field(
                    description="Der Elektrodenabstand kalt des Lechtmittels in Millimeter. Dazu der Such Operator, ob ein exaktes Match (==), größer (>), größer gleich (>=), kleiner gleich (<=) oder kleiner (<) als dieser Wert gesucht wird. Zum Beispiel (10.0, >=). Default zu exakt Match Operator. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                produktgewicht: Union[FloatSearchOperator, None] = Field(
                    description="Das Produktgewicht des Lechtmittels in Gramm. Dazu der Such Operator, ob ein exaktes Match (==), größer (>), größer gleich (>=), kleiner gleich (<=) oder kleiner (<) als dieser Wert gesucht wird. Zum Beispiel (10.0, >=). Default zu exakt Match Operator. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                kabel_laenge: Union[FloatSearchOperator, None] = Field(
                    description="Die Kabellänge des Lechtmittels. Dazu der Such Operator, ob ein exaktes Match (==), größer (>), größer gleich (>=), kleiner gleich (<=) oder kleiner (<) als dieser Wert gesucht wird. Zum Beispiel (10.0, >=). Default zu exakt Match Operator. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                max_temp: Union[IntSearchOperator, None] = Field(
                    description="Die maximal zulässige Umgebungstemperatur Quetschung des Lechtmittels in Grad Celsius. Dazu der Such Operator, ob ein exaktes Match (==), größer (>), größer gleich (>=), kleiner gleich (<=) oder kleiner (<) als dieser Wert gesucht wird. Zum Beispiel (10, >=). Default zu exakt Match Operator. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                lifetime: Union[IntSearchOperator, None] = Field(
                    description="Die Lebensdauer des Lechtmittels in Stunden/h. Dazu der Such Operator, ob ein exaktes Match (==), größer (>), größer gleich (>=), kleiner gleich (<=) oder kleiner (<) als dieser Wert gesucht wird. Zum Beispiel (10, >=). Default zu exakt Match Operator. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                warranty: Union[IntSearchOperator, None] = Field(
                    description="Die Service Warranty Lifetime des Lechtmittels in Stunden/h. Dazu der Such Operator, ob ein exaktes Match (==), größer (>), größer gleich (>=), kleiner gleich (<=) oder kleiner (<) als dieser Wert gesucht wird. Zum Beispiel (10, >=). Default zu exakt Match Operator. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                sockel_anode: Union[str, None] = Field(
                    description="Die Sockel Anode (Normbezeichnung) des Lechtmittels. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                sockel_kathode: Union[str, None] = Field(
                    description="Die Sockel Kathode (Normbezeichnung) des Lechtmittels. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                anmerkung_produkt: Union[str, None] = Field(
                    description="Die Anmerkung zum Produkt des Lechtmittels. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                kuehlung: Union[str, None] = Field(
                    description="Die Kühlung des Lechtmittels. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                brennstellung: Union[str, None] = Field(
                    description="Die Brennstellung des Lechtmittels. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                datum_deklaration: Union[DateSearchOperator, None] = Field(
                    description="Das Datum der Deklaration des Lechtmittels im DD-MMM-YYYY Format. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                erzeugnisnummer: Union[int, None] = Field(
                    description="Die Erzeugnisnummer des Lechtmittels. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                stoff_kandidatenliste: Union[str, None] = Field(
                    description="Der Stoff der Kandidatenliste des Lechtmittels. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                stoff_cas_nr: Union[str, None] = Field(
                    description="Die CAS Nr. des Stoffes des Lechtmittels. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                info_sicherer_gebrauch: Union[str, None] = Field(
                    description="Die nformationen zum sicheren Gebrauch des Lechtmittels. Falls der Nutzer nichts derartiges fragt, extrahiere None.")
                scip_deklarationsnummer: Union[str, None] = Field(
                    description="Die SCIP Deklarationsnummer des Lechtmittels. Falls der Nutzer nichts derartiges fragt, extrahiere None.")

            response = self.llm_client.chat.completions.create(
                model=self.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                response_model=ExtractData,
                temperature=0
            )

            state['retrieved_data'] = filter_dataframe(df=state['data'], filters=response.dict())

            return state

        def generate_response(state: AgentState) -> AgentState:
            messages = state['messages']

            system_message = (
                "Du bist ein sachkundiger und freundlicher Assistent der Informationen über Leuchtmittel bereitstellt. "
                "Antworte immer auf Deutsch. ")

            if state['chat_mode'] == "retrieval":
                system_message += ("Beantworte die Nutzeranfrage mit den folgenden Retrieval Ergebnissen:. "
                                   f"Ergebnisse: " + str(state['retrieved_data']) +
                                   " Falls die Ergebnisse None sind, sage dem Nutzer dass keine Leuctmittel zu dieser Anfrage gefunden wurden.")

            # Prepare the messages for the API call
            formatted_messages = [{"role": "system", "content": system_message}] + [
                {"role": "assistant" if isinstance(message, AIMessage) else "user", "content": message.content}
                for message in messages
            ]

            # Call the OpenAI API with the latest method for generating completions
            client = OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(model=self.llm_model_name,
                                                      messages=formatted_messages,
                                                      temperature=0)

            # Update the state with the response content
            state['response'] = response.choices[0].message.content
            return state

        def routing_function(state: AgentState):
            if state['chat_mode'] == "retrieval":
                return True
            else:
                return False

        workflow.add_node("select_chat_mode", select_chat_mode)
        workflow.add_node("generate_response", generate_response)
        workflow.add_node("retrieve", retrieve)

        workflow.set_entry_point("select_chat_mode")
        # Add conditional edges
        workflow.add_conditional_edges(
            "select_chat_mode",
            routing_function,
            {True: "retrieve", False: "generate_response"}
        )
        workflow.add_edge("retrieve", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    def chat(self, messages: List[Dict[str, str]]) -> str:
        # Update the state with new messages
        self.state['messages'] = [
            AIMessage(content=m["content"]) if m["role"] == "assistant" else HumanMessage(content=m["content"])
            for m in messages
        ]

        final_state = self.workflow.invoke(self.state)
        # print("Final state after workflow invocation:", final_state)  # Print final state for debugging
        return final_state['response']
