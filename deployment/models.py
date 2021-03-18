from django.db import models


class Deployment(models.Model):

	def __init__(self, sentence):
		self.sentence = sentence
	def token(self):
		return self.sentence.split(' ')