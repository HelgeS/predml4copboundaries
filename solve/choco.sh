#!/bin/sh
# Call: ./choco.sh <MZN> <DZN> <Timeout> [<Path to choco-parsers>]
MZN=$1
DZN=$2
TIMEOUT=$3
CHOCODIR="/home/helge/Sandbox/choco-parsers"

MZNFILE=`basename ${MZN}`
FZN="/tmp/choco-${MZNFILE}.fzn"

mzn2fzn -I "${CHOCODIR}/src/chocofzn/globals" ${MZN} -d ${DZN} -o ${FZN}

sh ${CHOCODIR}/src/chocofzn/fzn-exec -jar ${CHOCODIR}/choco-parsers-4.0.5-with-dependencies.jar -tl ${TIMEOUT} -a ${FZN}

rm ${FZN}
