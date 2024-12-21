NAMES=(0 1 2 3)
PERMS=(0 1 2 3 4 5 6 7 8 9)

for NAME in ${NAMES[@]}; do
    for PERM in ${PERMS[@]}; do
        eval "python synthetic_JVAE.py ${NAME} ${PERM}"
        eval "python synthetic_CVAE.py ${NAME} ${PERM}"
        eval "python synthetic_GMM.py ${NAME} ${PERM}"
        eval "python synthetic_RMPP.py ${NAME} ${PERM}"
    done
done
