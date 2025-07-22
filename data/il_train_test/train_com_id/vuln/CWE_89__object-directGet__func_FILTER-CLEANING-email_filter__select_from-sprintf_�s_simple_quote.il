class class_name1
  public function function_name1
    return GET
var1 new class_name1
var2 var1->function_name1
var3 filter_var var2 FILTER_SANITIZE_EMAIL
if filter_var var3 FILTER_VALIDATE_EMAIL
  var2 var3
else
  var2
var4 sprintf SQL_QUERY var2
var5 mysql_connect
mysql_select_db
echo var4
var6 mysql_query var4
loop var7 mysql_fetch_array var6
  print_r var7
  echo
mysql_close var5
