# coding=utf-8
"""
PAT - the name of the current project.
publisher.py - the name of the new file which you specify in the New File
dialog box during the file creation.
Hossein - the login name of the current user.
8 / 6 / 18 - the current system date.
8: 03 PM - the current system time.
PyCharm - the name of the IDE in which the file will be created.
"""
# import time
import pickle
import logging

from google.cloud import pubsub_v1


def list_topics(project):
    """Lists all Pub/Sub topics in the given project."""
    # [START pubsub_list_topics]
    publisher = pubsub_v1.PublisherClient()
    project_path = publisher.project_path(project)

    topics = [topic for topic in publisher.list_topics(project_path)]
    return topics


def create_topic(project, topic_name):
    """Create a new Pub/Sub topic."""
    # [START pubsub_create_topic]
    topics = list_topics(project)
    if topic_name in str(topics):
        delete_topic(project, topic_name)
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project, topic_name)

    topic = publisher.create_topic(topic_path)

    print('Topic created: {}'.format(topic))
    # [END pubsub_create_topic]


def delete_topic(project, topic_name):
    """Deletes an existing Pub/Sub topic."""
    # [START pubsub_delete_topic]
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project, topic_name)

    publisher.delete_topic(topic_path)

    print('Topic deleted: {}'.format(topic_path))


def delete_all_topics(project):
    """Deletes all existing Pub/Sub topics."""
    topics = list_topics(project)
    topics = [str(topic).split('/')[-1][:-2] for topic in topics]
    for topic_name in topics:
        delete_topic(project, topic_name)
        print('Topic deleted: {}'.format(topic_name))
