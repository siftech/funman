#!/bin/bash

pip install datamodel-code-generator[http] || echo "Didn't install datamodel-code-generator (already installed?)"
# datamodel-codegen --reuse-model --url https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/petrinet/petrinet_schema.json --input-file-type jsonschema --output-model-type pydantic_v2.BaseModel --output src/funman/model/generated_models/petrinet.py 
# datamodel-codegen --reuse-model --url https://raw.githubusercontent.com/DARPA-ASKEM/Model-Representations/main/regnet/regnet_schema.json --input-file-type jsonschema --output-model-type pydantic_v2.BaseModel --output src/funman/model/generated_models/regnet.py

datamodel-codegen --reuse-model --input resources/amr/petrinet_schema.json --input-file-type jsonschema --output-model-type pydantic_v2.BaseModel --output src/funman/model/generated_models/petrinet.py 
datamodel-codegen --reuse-model --input resources/amr/regnet_schema.json --input-file-type jsonschema --output-model-type pydantic_v2.BaseModel --output src/funman/model/generated_models/regnet.py