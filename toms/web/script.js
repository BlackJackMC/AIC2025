const config = JSON.parse(config_json.textContent)
const fps_map = new Map(Object.entries(JSON.parse(fps_json.textContent)))

search_form.addEventListener("submit", async e => {
  e.preventDefault()

  const data = new FormData(e.target)
  document.title = "⏳ Loading"
  const response = await fetch("/search", {
    method: "POST",
    body: data.get("query")
  })

  result_container.innerHTML = await response.text()
  document.title = "Gawr Gura video browser"
})

clear_form.addEventListener("submit", e => {
  e.preventDefault()

  const data = new FormData(e.target)
  const video_id = data.get("video")
  if (video_id == "") {
    for (const e of document.querySelectorAll(".result-item")) {
      e.classList.remove("hidden")
    }
    return
  }

  for (const e of document.querySelectorAll(`[data-video-id=${video_id}]`)) {
    e.classList.toggle("hidden")
  }
})

login_button.addEventListener("click", async () => {
  submit_log.textContent = "logging in..."
  const res = (await (await fetch(`${config.api_url}/login`, {
    method: "POST",
    body: JSON.stringify({
      username: config.username,
      password: config.password
    })
  })).json())
  submit_log.innerHTML = JSON.stringify(res, null, 2)
  session_field.value = res.sessionId
})

submit_form.addEventListener("submit", async e => {
  e.preventDefault()
  submit_log.textContent = "submitting..."
  const res = (await (await fetch(`${config.api_url}/client/evaluation/list?session=${session_field.value}`, {
    method: "GET"
  })).json())
  config.id = res[0].id
  e.preventDefault()
  submit_log.textContent = "submitting..."
  const data = new FormData(e.target)
  const video = data.get("video")
  const time = data.get("time")
  const QA = data.get("QA")
  submit_log.textContent = `${video} ${time} ${QA}`
  if(QA != ''){
    const res2 = (await (await fetch(`${config.api_url}/submit/${config.id}?session=${session_field.value}`, {
      method: "POST",
      body: JSON.stringify({
        "answerSets":[{
          "answers": [{
            "text": `${QA}-${video}-${time}`,
          }]
        }],
      }),
    })).json())
    submit_log.textContent = JSON.stringify(res2, null, 2)
  } else{
    const res2 = (await (await fetch(`${config.api_url}/submit/${config.id}?session=${session_field.value}`, {
      method: "POST",
      body: JSON.stringify({
        "answerSets":[{
          "answers": [{
            "mediaItemName": video,
            "start": time,
            "end": time
          }]
        }],
      }),
    })).json())
    submit_log.textContent = JSON.stringify(res2, null, 2)
  }
})

