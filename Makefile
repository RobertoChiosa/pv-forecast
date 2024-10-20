define find.functions
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
endef

help:
	@echo 'The following commands can be used.'
	@echo ''
	$(call find.functions)



.PHONY: setup
update: ## Update dependency of the project
setup:
	@echo "Updating dependencies"
	source .venv/bin/activate && \
	pip install poetry && \
	poetry install


.PHONY: rm-git-cache
rm-git-cache: ## Remove git cached files
rm-git-cache:
	@echo "Removing git cached files"
	git rm -r --cached .
	git add .


.PHONY: format
format: ## Format according to black and isort
format:
	@echo "Formatting..."
	source .venv/bin/activate && \
	black . && \
	isort .