var1 fopen
if var1
  if var2 fgets var1 4096 false
    var2
  fclose var1
else
  var2
if filter_var var3 FILTER_VALIDATE_EMAIL
  var2 var3
else
  var2
var4 SQL_QUERY var2
var5 mysql_connect
mysql_select_db
echo var4
var6 mysql_query var4
loop var7 mysql_fetch_array var6
  print_r var7
  echo
mysql_close var5
