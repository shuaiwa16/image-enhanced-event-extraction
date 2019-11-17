PPPP="hahah<sd>"
commandOutput="$(echo $PPPP)"
echo $commandOutput


str="this is a string"
[[ $str =~ "this" ]] && echo "$str contains this" 
[[ $str =~ "that" ]] || echo "$str does NOT contain that"

PPPP="hahah<sd>"
PATTERN="hah"
# result=$(echo $PPPP | grep "${PATTERN}")
if [[ "$PPPP" =~ "$PATTERN" ]]
then
	echo "right"
else
	echo "wrong"
fi


arr=(2 3 3)
for var in ${arr[@]};
do
    echo $var
done
