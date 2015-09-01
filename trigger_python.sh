body='{
"request": {
  "branch":"master"
}}'

curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Travis-API-Version: 3" \
  -H "Authorization: token $1" \
  -d "$body" \
  https://api.travis-ci.org/repo/libdynd%2Fdynd-python/requests
