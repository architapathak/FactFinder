from django import template

register = template.Library()

def cut(value, arg):

	return value.split(' ')