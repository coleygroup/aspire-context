FROM tensorflow/serving:latest

COPY --chown=tensorflow:tensorflow ./models/context/ /models/context/

COPY --chown=tensorflow:tensorflow ./tfx/models.config /models/

CMD [ "--model_config_file=/models/models.config" ]
