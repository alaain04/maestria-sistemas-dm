TOPIC_NAME=orders-topic
/opt/kafka/bin/kafka-topics.sh  --create \
                                --zookeeper zookeeper:2181 \
                                --replication-factor 1 \
                                --partitions 1 \
                                --topic $TOPIC_NAME