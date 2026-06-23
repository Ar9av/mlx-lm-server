use axum::response::Html;

pub async fn chat_ui() -> Html<&'static str> {
    Html(include_str!("ui.html"))
}
