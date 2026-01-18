files=()
for i in $(seq -w 0 20); do
  f="lec${i}/lec${i}.pdf"
  [ -f "$f" ] && files+=("$f")
done

gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=all_lectures.pdf "${files[@]}"

