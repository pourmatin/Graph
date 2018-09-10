# coding=utf-8
"""
PAT - the name of the current project.
subscriber.py - the name of the new file which you specify in the New File
dialog box during the file creation.
Hossein - the login name of the current user.
8 / 6 / 18 - the current system date.
10: 03 PM - the current system time.
PyCharm - the name of the IDE in which the file will be created.
"""
from google.cloud import pubsub_v1


def list_subscriptions_in_topic(project, topic_name):
    """Lists all subscriptions for a given topic."""
    # [START pubsub_list_topic_subscriptions]
    subscriber = pubsub_v1.PublisherClient()
    topic_path = subscriber.topic_path(project, topic_name)

    for subscription in subscriber.list_topic_subscriptions(topic_path):
        print(subscription)


def list_subscriptions_in_project(project):
    """Lists all subscriptions in the current project."""
    # [START pubsub_list_subscriptions]
    subscriber = pubsub_v1.SubscriberClient()
    project_path = subscriber.project_path(project)

    subscriptions = [subscription.name for subscription in subscriber.list_subscriptions(project_path)]
    return subscriptions


def create_subscription(project, topic_name, subscription_name):
    """Create a new pull subscription on the given topic."""
    # [START pubsub_create_pull_subscription]
    subscriptions = list_subscriptions_in_project(project)
    subscriptions = [subscription.split('/')[-1] for subscription in subscriptions]
    if subscription_name in subscriptions:
        delete_subscription(project, subscription_name)
    subscriber = pubsub_v1.SubscriberClient()
    topic_path = subscriber.topic_path(project, topic_name)
    subscription_path = subscriber.subscription_path(
        project, subscription_name)

    subscription = subscriber.create_subscription(
        subscription_path, topic_path)

    print('Subscription created: {}'.format(subscription))


def delete_subscription(project, subscription_name):
    """Deletes an existing Pub/Sub topic."""
    # [START pubsub_delete_subscription]
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(
        project, subscription_name)

    subscriber.delete_subscription(subscription_path)

    print('Subscription deleted: {}'.format(subscription_path))
    # [END pubsub_delete_subscription]


def update_subscription(project, subscription_name, endpoint):
    """
    Updates an existing Pub/Sub subscription's push endpoint URL.
    Note that certain properties of a subscription, such as
    its topic, are not modifiable.
    """
    # [START pubsub_update_push_configuration]
    endpoint = endpoint or "https://{0}.appspot.com/push".format(project)
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(
        project, subscription_name)

    push_config = pubsub_v1.types.PushConfig(
        push_endpoint=endpoint)

    subscription = pubsub_v1.types.Subscription(
        name=subscription_path,
        push_config=push_config)

    update_mask = {
        'paths': {
            'push_config',
        }
    }

    subscriber.update_subscription(subscription, update_mask)
    result = subscriber.get_subscription(subscription_path)

    print('Subscription updated: {}'.format(subscription_path))
    print('New endpoint for subscription is: {}'.format(
        result.push_config))
    # [END pubsub_update_push_configuration]
