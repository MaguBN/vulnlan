var1 shell_exec
class class_name1
  public function function_name1 var2
    return mysql_real_escape_string var2
var3 new class_name1
var1 var3->function_name1 var1
var4 sprintf SQL_QUERY var1
var5 mysql_connect
mysql_select_db
echo var4
var6 mysql_query var4
loop var7 mysql_fetch_array var6
  print_r var7
  echo
mysql_close var5
