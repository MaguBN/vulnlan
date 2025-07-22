var1 array
0 array
1 array
2 array
var2
var3 proc_open var1 var4 var2 NULL
if is_resource var3
  fclose var4_0
  var5 stream_get_contents var4_1
  fclose var4_1
  var6 proc_close var3
var7 filter_var var5 FILTER_SANITIZE_SPECIAL_CHARS
var5 var7
var8 SQL_QUERY var5
var9 mysql_connect
mysql_select_db
echo var8
var10 mysql_query var8
loop var11 mysql_fetch_array var10
  print_r var11
  echo
mysql_close var9
