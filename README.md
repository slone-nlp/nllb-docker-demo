

This is an example Dockerized application that serves a machine translation model.

It is described in the post
https://medium.com/@cointegrated/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865

## Prerequisites
Docker. A few gigabytes of memory.

## How to run

Build a Docker image called "nllb" (from the current directory): 
```
docker build -t nllb .
```

Run it: 
```
docker run  -it -p 7860:7860 nllb
```

Now open the browser at http://localhost:7860/docs. 
It will show you a signature of the method you can use for translation.

## How to adapt
If you want to deploy another NLLB-based translation model, 
just change the `MODEL_URL` in the `translation.py` file.
You may also want to adjust the `LANGUAGES` register in the same file.


