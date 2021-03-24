#!/usr/bin/env bash

if [ "${TRAVIS_COMMIT_RANGE}" ]
then
	COMMIT_RANGE="${TRAVIS_COMMIT_RANGE/.../ }"
	echo "Checking formatting for commit range: ${COMMIT_RANGE}"
	cd "${TRAVIS_BUILD_DIR}"
	DIFF="$(./scripts/developer/git/git-clang-format --diff ${COMMIT_RANGE})"
	GCA_EXIT_CODE=$?
	if [ ${GCA_EXIT_CODE} -ne 0 ]
	then
		exit ${GCA_EXIT_CODE}
	fi
	if [ "${DIFF}" ] && [ "${DIFF}" != "no modified files to format" ]
	then
		echo "clang-format:"
		echo "  Code format checks failed."
		echo "  Please run clang-format (or git clang-format) on your changes"
		echo "  before committing."
		echo "  The following changes are suggested:"
		echo "${DIFF}"
		exit 1
	fi
fi

exit 0
