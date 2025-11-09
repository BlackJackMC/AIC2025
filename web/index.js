function debounce(func, delay) {
    let timer;

    return function () {
        const args = arguments;
        const scope = this;
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => { func.apply(scope, args); }, delay);
    };
}

const dres_url = "https://eventretrieval.oj.io.vn/api/v2"
const backend_url = "http://38.29.145.105:1111"
const login_form = document.querySelector("#login")
const submission_form = document.querySelector("#submission")
const get_eval_button = document.querySelector("#get_eval");
const evaluation_select = document.querySelector("#submission select[name=evaluation]");
const mode_select = document.querySelector("#submission select[name=mode]");

function getSessionId() {
    return sessionStorage.getItem("session")
}

function isLoggedIn() {
    let sessionId = getSessionId()
    return sessionId ? true : false;
}

function updateLoginStatus() {
    const statusElem = document.querySelector("#login div.status")

    if (!isLoggedIn()) {
        statusElem.innerHTML = "Not logged in";
    } else {
        statusElem.innerHTML = "Logged in";
    }
}

function saveForm(form) {
    const data = Object.fromEntries(new FormData(form));
    localStorage.setItem(`formData:${form.id}`, JSON.stringify(data));
}

function loadForm(form) {
    const saved = localStorage.getItem(`formData:${form.id}`);
    if (!saved) return;
    const data = JSON.parse(saved);
    for (const [key, value] of Object.entries(data)) {
        const input = form.elements[key];
        if (input) input.value = value;
    }
}

async function updateEvaluationId() {
    const response = await fetch(`${dres_url}/client/evaluation/list?session=${getSessionId()}`, {
        method: "GET"
    });

    if (response.ok) {
        const data = await response.json();
        console.log(data);

        list = "<option disabled selected hidden>Choose evaluation id</option>"

        data.forEach((evaluation) => {
            list += `
            <option>${evaluation.id}</option>`
        });

        evaluation_select.innerHTML = list;
    }
}

document.addEventListener("DOMContentLoaded", async () => {
    const forms = document.querySelectorAll("form[id]");

    forms.forEach((form) => {
        loadForm(form);
        form.addEventListener("input", () => saveForm(form));
    });

    updateLoginStatus();
    await updateEvaluationId();
});

login_form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const formData = new FormData(login_form);
    const username = formData.get("username");
    const password = formData.get("password");

    const response = await fetch(`${dres_url}/login`, {
        method: "POST",
        body: JSON.stringify({
            "username": username,
            "password": password
        })
    });

    const responseData = await response.json();

    console.log(responseData);
    if (response.ok && responseData["sessionId"]) {
        sessionStorage.setItem("session", responseData["sessionId"])
    } else {
        sessionStorage.removeItem("session");
    }

    updateLoginStatus();
});

get_eval_button.addEventListener("click", async (event) => {
    event.preventDefault();

    updateEvaluationId();
});