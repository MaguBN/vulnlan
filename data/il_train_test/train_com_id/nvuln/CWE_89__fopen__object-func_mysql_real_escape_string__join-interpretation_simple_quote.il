var1 fopen
if var1
  if var2 fgets var1 4096 false
    var2
  fclose var1
else
  var2
class class_name1
  public function function_name1 var3
    return mysql_real_escape_string var3
var4 new class_name1
var2 var4->function_name1 var2
var5 SQL_QUERY var2
var6 mysql_connect
mysql_select_db
echo var5
var7 mysql_query var5
loop var8 mysql_fetch_array var7
  print_r var8
  echo
mysql_close var6
