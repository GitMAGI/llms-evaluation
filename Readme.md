# Large Language Models Evaluation

_Note: It seems to be not possible to mount a volume in windows specifying disk letter. So if you want to mount a volume with absolute path, you have to run the docker-compose command from the same disk you want to mount as volume_

## OLLAMA <a href="https://medium.com/@edu.ukulelekim/how-to-locally-deploy-ollama-and-open-webui-with-docker-compose-318f0582e01f">:link:</a>

### Include a model (from ollama.com) 
- navigate to http://localhost:3011/admin/settings
- click on the wrench
- you need to put the exact name and tag of the model you want to add; these names are listed <a href="https://ollama.com/library">here</a> 