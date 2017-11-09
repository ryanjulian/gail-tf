here="$( cd "$(dirname "$0")" ; pwd -P )"
export PYTHONPATH="${here}:${PYTHONPATH}"
echo "PYTHONPATH set to ${PYTHONPATH}"
