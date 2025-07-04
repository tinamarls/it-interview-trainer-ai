import React, { useState, useRef } from "react";

export const StressTrain = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [errorMessage, setErrorMessage] = useState("");
    const videoRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const recordedChunks = useRef([]);

    // Константа для questionId
    const QUESTION_ID = 1; // Замените на нужный questionId

    const startRecording = async () => {
        setErrorMessage("");
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
            videoRef.current.srcObject = stream;
            videoRef.current.play();

            mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: "video/webm" });
            mediaRecorderRef.current.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    recordedChunks.current.push(event.data);
                }
            };
            mediaRecorderRef.current.start();
            setIsRecording(true);
        } catch (error) {
            setErrorMessage("Не удалось получить доступ к веб-камере.");
            console.error("Ошибка:", error);
        }
    };

    const stopRecording = async () => {
        mediaRecorderRef.current.stop();
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
        setIsRecording(false);

        const blob = new Blob(recordedChunks.current, { type: "video/webm" });
        recordedChunks.current = [];

        // Отправка видео на бэкенд
        const formData = new FormData();
        formData.append("video", blob);
        formData.append("questionId", QUESTION_ID); // Добавляем questionId в formData

        try {
            const response = await fetch(`${process.env.REACT_APP_API_URL}/video/upload`, {
                method: "POST",
                body: formData,
            });

            if (response.ok) {
                alert("Видео успешно отправлено!");
            } else {
                setErrorMessage("Не удалось отправить видео. Попробуйте снова.");
            }
        } catch (error) {
            setErrorMessage("Произошла ошибка при отправке видео.");
            console.error("Ошибка:", error);
        }
    };

    return (
        <div className="content">
            <div className="video-response">
                <h1>Расскажите о вашем опыте работы</h1>
                <div className="video-container">
                    <video ref={videoRef} className="video-preview" />
                </div>
                <div className="controls">
                    {!isRecording ? (
                        <button onClick={startRecording}>Начать запись</button>
                    ) : (
                        <button onClick={stopRecording}>Ответить</button>
                    )}
                </div>
                {errorMessage && <p className="error-message">{errorMessage}</p>}
            </div>
        </div>
    );
};
