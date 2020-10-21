# -*- coding: utf-8 -*-
from kafka import KafkaConsumer, KafkaProducer
from json import dumps

# =====================================================================
# ------------------------- Kafka Connector ---------------------------
# =====================================================================


class KafkaConnector:
    def __init__(self, topic_consumer=None, topic_producer=None, group_id="parking-group",
                 bootstrap_servers=None, enable_auto_commit=False,
                 consumer_timeout_ms=1000, auto_offset_reset="earliest"):

        self.topic_consumer = topic_consumer
        self.topic_producer = topic_producer
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers
        self.enable_auto_commit = enable_auto_commit
        self.consumer_timeout_ms = consumer_timeout_ms
        self.auto_offset_reset = auto_offset_reset
        self.consumer = None
        self.producer = None

    def init_kafka_consumer(self):
        try:
            """Init Consumer which is in charge of reading the information from a Kafka topic queue
            """
            self.consumer = KafkaConsumer(self.topic_consumer,
                                          group_id=self.group_id,
                                          bootstrap_servers=self.bootstrap_servers,
                                          auto_offset_reset=self.auto_offset_reset,
                                          enable_auto_commit=self.enable_auto_commit)
        except Exception as e:
            print(e)

    def init_kafka_producer(self):
        try:
            """Init Consumer which is in charge of writing the information into a Kafka topic queue
            """
            self.producer = KafkaProducer(bootstrap_servers=self.bootstrap_servers,
                                          value_serializer=lambda x: dumps(x).encode('utf-8'))
        except Exception as e:
            print(e)

    def put_data_into_topic(self, data):
        try:
            if self.producer is not None:
                self.producer.send(topic=self.topic_producer, value=data)
        except Exception as e:
            print(e)


