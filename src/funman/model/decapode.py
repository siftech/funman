from typing import Dict, List, Union

from pydantic import BaseModel

from .model import FunmanModel


class DecapodeDynamics(BaseModel):
    json_graph: Dict[str, List[Dict[str, Union[int, str]]]]


class DecapodeModel(FunmanModel):
    decapode: DecapodeDynamics

    def default_encoder(
        self, config: "FUNMANConfig", scenario: "AnalysisScenario"
    ) -> "Encoder":
        """
        Return the default Encoder for the model

        Returns
        -------
        Encoder
            SMT encoder for model
        """
        from funman.translate.decapode import DecapodeEncoder

        return DecapodeEncoder(config=config, scenario=scenario)
