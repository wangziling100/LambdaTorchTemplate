build-SamDemoFunction:
	@echo "Buliding artifacts with sls. Destination dir " $(ARTIFACTS_DIR)
	mv dist/torchlambda.zip $(ARTIFACTS_DIR)
	unzip $(ARTIFACTS_DIR)/torchlambda.zip -d $(ARTIFACTS_DIR)
	rm $(ARTIFACTS_DIR)/torchlambda.zip
build-ModelLayer:
	mv dist/model.zip $(ARTIFACTS_DIR)
	unzip $(ARTIFACTS_DIR)/model.zip -d $(ARTIFACTS_DIR)
	rm $(ARTIFACTS_DIR)/model.zip