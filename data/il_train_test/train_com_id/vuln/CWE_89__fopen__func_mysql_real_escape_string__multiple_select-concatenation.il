var1 fopen
if var1
  if var2 fgets var1 4096 false
    var2
  fclose var1
else
  var2
var2 mysql_real_escape_string var2
var3 SQL_QUERY var2
var4 mysql_connect
mysql_select_db
echo var3
var5 mysql_query var3
loop var6 mysql_fetch_array var5
  print_r var6
  echo
mysql_close var4
