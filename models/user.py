from mongoengine import Document, StringField

class NewUser(Document):
    username = StringField(required=True, unique=True)
    email = StringField(required=True, unique=True)
    password = StringField(required=True)