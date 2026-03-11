
clean_dist:
	rm -rf dist/*

create_dist: clean_dist
	python setup.py sdist

upload_package: create_dist
	twine upload dist/*

check_workflow_triggers:
	python3 scripts/check_no_push_triggers.py
