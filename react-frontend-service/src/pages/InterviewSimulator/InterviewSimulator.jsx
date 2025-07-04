// import React, {useState, useRef, useEffect, useContext} from "react";
// import {useNavigate} from "react-router-dom";
// import "./InterviewSimulator.css";
// import {fetchWithAuth} from "../../api/fetchWithAuth";
// import {AuthContext} from "../AuthProvider";
//
// export const InterviewSimulator = () => {
//     const navigate = useNavigate();
//     const [isRecording, setIsRecording] = useState(false);
//     const [errorMessage, setErrorMessage] = useState("");
//     const [timeLeft, setTimeLeft] = useState(90); // 1.5 минуты = 90 секунд
//     const [isInterviewStarted, setIsInterviewStarted] = useState(false);
//     const [isInterviewFinished, setIsInterviewFinished] = useState(false);
//     const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
//     const [answeredQuestions, setAnsweredQuestions] = useState([]);
//     const [attemptId, setAttemptId] = useState(null); // ID попытки
//     const [isRecordingReady, setIsRecordingReady] = useState(false); // Новый флаг готовности записи
//     const {token} = useContext(AuthContext);
//
//     const videoRef = useRef(null);
//     const mediaRecorderRef = useRef(null);
//     const recordedChunks = useRef([]);
//     const timerRef = useRef(null);
//
//     const questions = [
//         {id: 1, text: "Расскажите о вашем опыте работы"},
//         {id: 2, text: "Почему вас заинтересовала эта вакансия?"},
//         {id: 3, text: "Какими технологиями вы владеете?"},
//         {id: 4, text: "Расскажите о сложном проекте, над которым вы работали"},
//         {id: 5, text: "Каковы ваши карьерные цели на ближайшие 3-5 лет?"}
//     ];
//
//     useEffect(() => {
//         // Очищаем таймер при размонтировании компонента
//         return () => {
//             if (timerRef.current) {
//                 clearInterval(timerRef.current);
//             }
//         };
//     }, []);
//
//     const startInterview = async () => {
//         try {
//             // Отправляем запрос на создание новой попытки HR-интервью
//             const response = await fetchWithAuth("/hr-interview",
//                 {
//                     method: "POST", headers: {
//                         'Content-Type': 'application/json'
//                     }
//                 }, token);
//
//             if (response.ok) {
//                 const data = await response.json();
//                 setAttemptId(data.id); // Сохраняем полученный ID попытки
//                 setIsInterviewStarted(true);
//             } else {
//                 const errorData = await response.json();
//                 setErrorMessage(`Ошибка при начале интервью: ${errorData.error || 'Неизвестная ошибка'}`);
//             }
//         } catch (error) {
//             setErrorMessage(`Произошла ошибка при начале интервью: ${error.message}`);
//             console.error("Ошибка при начале интервью:", error);
//         }
//     };
//
//     const startTimer = () => {
//         setTimeLeft(90);
//         timerRef.current = setInterval(() => {
//             setTimeLeft((prevTime) => {
//                 if (prevTime <= 1) {
//                     stopRecording();
//                     clearInterval(timerRef.current);
//                     return 0;
//                 }
//                 return prevTime - 1;
//             });
//         }, 1000);
//     };
//
//     const resetTimer = () => {
//         clearInterval(timerRef.current);
//         setTimeLeft(90);
//     };
//
//     const startRecording = async () => {
//         setErrorMessage("");
//         recordedChunks.current = []; // Очищаем предыдущие чанки
//         setIsRecordingReady(false); // Сбрасываем флаг готовности записи
//
//         try {
//             const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: true});
//             videoRef.current.srcObject = stream;
//             videoRef.current.play();
//
//             mediaRecorderRef.current = new MediaRecorder(stream, {
//                 mimeType: "video/webm",
//                 videoBitsPerSecond: 2500000, // Примерное значение, подберите под свои нужды
//                 audioBitsPerSecond: 128000,  // Примерное значение
//             });
//             mediaRecorderRef.current.ondataavailable = (event) => {
//                 if (event.data.size > 0) {
//                     recordedChunks.current.push(event.data);
//                 }
//             };
//
//             // Добавляем обработчик для события start
//             mediaRecorderRef.current.onstart = () => {
//                 // Устанавливаем флаг готовности записи через небольшую задержку
//                 setTimeout(() => {
//                     setIsRecordingReady(true);
//                 }, 1000);
//             };
//
//             mediaRecorderRef.current.start(); // Запускаем запись без timeslice
//             setIsRecording(true);
//             startTimer();
//
//         } catch (error) {
//             setErrorMessage(`Произошла ошибка при начале записи: ${error.message}`);
//             console.error("Ошибка при начале записи:", error);
//         }
//     };
//
//     const resetMediaRecorder = () => {
//         setIsRecording(false);
//
//         // Остановка всех треков медиапотока
//         if (videoRef.current && videoRef.current.srcObject) {
//             const tracks = videoRef.current.srcObject.getTracks();
//             tracks.forEach(track => track.stop());
//         }
//
//         // Очистка видеоэлемента
//         if (videoRef.current) {
//             videoRef.current.srcObject = null;
//         }
//
//         recordedChunks.current = [];
//         setIsRecordingReady(false);
//     };
//
//     const stopRecording = async () => {
//         if (!isRecording) return;
//
//         try {
//             // Проверка готовности записи
//             if (!isRecordingReady) {
//                 setErrorMessage("Запись не была инициализирована должным образом. Пожалуйста, попробуйте ещё раз.");
//                 // resetMediaRecorder() будет вызван в finally, он также останавливает треки.
//                 return;
//             }
//
//             resetTimer();
//
//             // 1. Остановить треки исходного медиапотока
//             // Это важно сделать перед остановкой MediaRecorder
//             if (videoRef.current && videoRef.current.srcObject) {
//                 const stream = videoRef.current.srcObject;
//                 stream.getTracks().forEach(track => track.stop());
//             }
//
//             // 2. Остановить MediaRecorder
//             // Убедимся, что MediaRecorder все еще существует и находится в состоянии записи
//             if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
//                 const recordingStopped = new Promise((resolve) => {
//                     // Событие onstop вызывается после того, как все данные были обработаны и Blob готов
//                     mediaRecorderRef.current.onstop = () => {
//                         resolve();
//                     };
//                     mediaRecorderRef.current.stop();
//                 });
//
//                 // Ждем полного завершения записи и обработки данных MediaRecorder
//                 await recordingStopped;
//             }
//
//             // Проверка наличия данных перед отправкой
//             if (recordedChunks.current.length === 0) {
//                 setErrorMessage("Не удалось записать видео. Пожалуйста, попробуйте ещё раз.");
//                 // resetMediaRecorder() будет вызван в finally
//                 return;
//             }
//
//             const blob = new Blob(recordedChunks.current, {type: 'video/webm'});
//
//             // Проверка размера файла
//             if (blob.size < 1000) { // Минимум 1KB
//                 setErrorMessage("Записанное видео слишком короткое или пустое. Пожалуйста, попробуйте ещё раз.");
//                 // resetMediaRecorder() будет вызван в finally
//                 return;
//             }
//
//             // Теперь можно отправлять видео
//             await uploadVideo(blob);
//
//         } catch (error) {
//             setErrorMessage(`Произошла ошибка при остановке записи: ${error.message}`);
//             console.error("Ошибка при остановке записи:", error);
//         } finally {
//             // resetMediaRecorder() очистит состояние, остановит треки (если еще не остановлены)
//             // и очистит srcObject видеоэлемента.
//             resetMediaRecorder();
//         }
//     };
//
//     // const uploadVideo = async (blob) => {
//     //     try {
//     //         const formData = new FormData();
//     //         formData.append('video', blob, 'recorded-video.webm');
//     //         formData.append('questionId', questions[currentQuestionIndex].id);
//     //         formData.append('attemptId', attemptId);
//     //
//     //         console.log(`Отправка видео размером: ${blob.size} байт`);
//     //
//     //         const response = await fetchWithAuth("/hr-interview/upload-video",
//     //             { method: "POST", body: formData }, token);
//     //
//     //         if (response.ok) {
//     //             // Добавляем текущий вопрос в список отвеченных
//     //             setAnsweredQuestions(prev => [...prev, questions[currentQuestionIndex].id]);
//     //
//     //             // Если это был последний вопрос, завершаем интервью
//     //             if (currentQuestionIndex === questions.length - 1) {
//     //                 setIsInterviewFinished(true);
//     //             } else {
//     //                 // Автоматически переходим к следующему вопросу
//     //                 setCurrentQuestionIndex(prevIndex => prevIndex + 1);
//     //             }
//     //         } else {
//     //             const errorData = await response.json();
//     //             setErrorMessage(`Ошибка при загрузке видео: ${errorData.error || 'Неизвестная ошибка'}`);
//     //             console.error("Ошибка при загрузке видео:", errorData);
//     //         }
//     //     } catch (error) {
//     //         setErrorMessage(`Произошла ошибка при загрузке видео: ${error.message}`);
//     //         console.error("Ошибка при загрузке видео:", error);
//     //     }
//     // };
//
//     const uploadVideo = async (blob) => {
//         try {
//             const formData = new FormData();
//             formData.append('video', blob, 'recorded-video.webm');
//             formData.append('questionId', questions[currentQuestionIndex].id);
//             formData.append('attemptId', attemptId);
//
//             console.log(`Отправка видео размером: ${blob.size} байт`);
//
//             // Сразу добавляем текущий вопрос в список отвеченных
//             setAnsweredQuestions(prev => [...prev, questions[currentQuestionIndex].id]);
//
//             // Переходим к следующему вопросу сразу, не дожидаясь ответа от сервера
//             if (currentQuestionIndex === questions.length - 1) {
//                 setIsInterviewFinished(true);
//             } else {
//                 setCurrentQuestionIndex(prevIndex => prevIndex + 1);
//             }
//
//             // Отправляем запрос асинхронно, но не блокируем UI
//             fetchWithAuth("/hr-interview/upload-video",
//                 {method: "POST", body: formData}, token)
//                 .then(response => {
//                     if (!response.ok) {
//                         return response.json().then(errorData => {
//                             throw new Error(errorData.error || 'Ошибка при загрузке видео');
//                         });
//                     }
//                 })
//                 .catch(error => {
//                     console.error("Ошибка при загрузке видео:", error);
//                     // Показываем ошибку, но не меняем текущее состояние интерфейса
//                     setErrorMessage(`Произошла ошибка при загрузке видео: ${error.message}. Продолжайте отвечать на оставшиеся вопросы.`);
//                 });
//
//             console.log("Видео успешно отправлено на сервер");
//             console.log(formData)
//
//         } catch (error) {
//             setErrorMessage(`Произошла ошибка при подготовке видео: ${error.message}`);
//             console.error("Ошибка при подготовке видео:", error);
//         }
//     };
//
//     const finishInterview = () => {
//         navigate(`/hr-interview/results/${attemptId}`); // Редирект на страницу с результатами
//     };
//
//     // Добавляем функцию для перехода на страницу истории попыток
//     const viewHrInterviewHistory = () => {
//         navigate("/hr-interview/results");
//     };
//
//
//     // Стартовая страница
//     if (!isInterviewStarted) {
//         return (
//             <div className="start-page">
//                 <h1>Симулятор HR-скрининга</h1>
//                 <p>
//                     Этот тренажер поможет вам подготовиться к собеседованию.
//                     Вам будет предложено ответить на 5 вопросов с записью видео.
//                     На каждый ответ у вас будет 1,5 минуты.
//                 </p>
//                 <div className="rules">
//                     <h2>Правила:</h2>
//                     <ul>
//                         <li>Ответьте на все 5 вопросов</li>
//                         <li>Максимальное время ответа - 1,5 минуты</li>
//                         <li>Говорите четко и уверенно</li>
//                         <li>После завершения вы получите результаты</li>
//                     </ul>
//                 </div>
//                 <button className="start-button" onClick={startInterview}>
//                     Начать HR-собеседование
//                 </button>
//                 {errorMessage && <p className="error-message">{errorMessage}</p>}
//
//                 {/* Добавляем кнопку просмотра истории попыток */}
//                 <div className="action-buttons">
//                     {!isInterviewStarted && (
//                         <button
//                             className="history-button"
//                             onClick={viewHrInterviewHistory}
//                         >
//                             Посмотреть историю попыток прохождения HR-скрининга
//                         </button>
//                     )}
//                 </div>
//             </div>
//         );
//     }
//
//     // Страница с результатами интервью
//     if (isInterviewFinished) {
//         return (
//             <div className="results-page">
//                 <h1>Собеседование завершено!</h1>
//                 <p>Спасибо за прохождение тренировочного собеседования.</p>
//                 <p>Ваши ответы были успешно записаны и отправлены на обработку.</p>
//                 <p>Вскоре вы получите результаты и рекомендации для улучшения.</p>
//                 <button className="finish-button" onClick={finishInterview}>
//                     Перейти к результатам
//                 </button>
//             </div>
//         );
//     }
//
//     // Основная страница симулятора интервью
//     return (
//         <div className="interview-container">
//             <h1>HR-собеседование</h1>
//             <div className="interview-layout">
//                 <div className="video-section">
//                     <div className="video-container">
//                         <video ref={videoRef} className="video-preview" muted/>
//                         {isRecording && (
//                             <div className="timer">
//                                 {Math.floor(timeLeft / 60)}:{(timeLeft % 60).toString().padStart(2, '0')}
//                             </div>
//                         )}
//                     </div>
//
//                     {/* Прогресс-бар с чётко заданным горизонтальным направлением */}
//                     <div className="progress-bar">
//                         {questions.map((q, index) => (
//                             <div
//                                 key={q.id}
//                                 className={`progress-item ${index === currentQuestionIndex ? 'current' : ''} ${answeredQuestions.includes(q.id) ? 'answered' : ''}`}
//                             >
//                                 {index + 1}
//                             </div>
//                         ))}
//                     </div>
//
//                     <div className="controls">
//                         {!isRecording ? (
//                             <button
//                                 className="record-button"
//                                 onClick={startRecording}
//                                 disabled={answeredQuestions.includes(questions[currentQuestionIndex].id)}
//                             >
//                                 Начать запись
//                             </button>
//                         ) : (
//                             <button
//                                 className="answer-button"
//                                 onClick={stopRecording}
//                             >
//                                 {currentQuestionIndex === questions.length - 1 ? "Завершить интервью" : "Ответить"}
//                             </button>
//                         )}
//                     </div>
//                 </div>
//
//                 <div className="question-section">
//                     <div className="question-card">
//                         <h3>Вопрос {currentQuestionIndex + 1} из {questions.length}</h3>
//                         <p className="question-text">{questions[currentQuestionIndex].text}</p>
//                     </div>
//                 </div>
//             </div>
//
//             {errorMessage && <p className="error-message">{errorMessage}</p>}
//         </div>
//     );
// };
//
// export default InterviewSimulator;

import React, {useState, useRef, useEffect, useContext} from "react";
import {useNavigate} from "react-router-dom";
import "./InterviewSimulator.css";
import {fetchWithAuth} from "../../api/fetchWithAuth";
import {AuthContext} from "../AuthProvider";

export const InterviewSimulator = () => {
    const navigate = useNavigate();
    const [isRecording, setIsRecording] = useState(false);
    const [errorMessage, setErrorMessage] = useState("");
    const [timeLeft, setTimeLeft] = useState(90); // 1.5 минуты = 90 секунд
    const [isInterviewStarted, setIsInterviewStarted] = useState(false);
    const [isInterviewFinished, setIsInterviewFinished] = useState(false);
    const [attemptId, setAttemptId] = useState(null);
    const [isRecordingReady, setIsRecordingReady] = useState(false);
    const {token} = useContext(AuthContext);

    const videoRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const recordedChunks = useRef([]);
    const timerRef = useRef(null);

    // Только один вопрос для stress-режима
    const question = {
        id: 1,
        text: "Расскажите о вашем прошлом месте работы"
    };

    useEffect(() => {
        return () => {
            if (timerRef.current) {
                clearInterval(timerRef.current);
            }
        };
    }, []);

    const startInterview = async () => {
        try {
            const response = await fetchWithAuth("/hr-interview",
                {
                    method: "POST",
                    headers: {
                        'Content-Type': 'application/json'
                    }
                },
                token
            );

            if (response.ok) {
                const data = await response.json();
                setAttemptId(data.id);
                setIsInterviewStarted(true);
            } else {
                const errorData = await response.json();
                setErrorMessage(`Ошибка при начале интервью: ${errorData.error || 'Неизвестная ошибка'}`);
            }
        } catch (error) {
            setErrorMessage(`Произошла ошибка при начале интервью: ${error.message}`);
            console.error("Ошибка при начале интервью:", error);
        }
    };

    const startTimer = () => {
        setTimeLeft(90);
        timerRef.current = setInterval(() => {
            setTimeLeft((prevTime) => {
                if (prevTime <= 1) {
                    stopRecording();
                    clearInterval(timerRef.current);
                    return 0;
                }
                return prevTime - 1;
            });
        }, 1000);
    };

    const startRecording = async () => {
        setErrorMessage("");
        recordedChunks.current = [];
        setIsRecordingReady(false);

        try {
            const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: true});
            videoRef.current.srcObject = stream;
            videoRef.current.play();

            mediaRecorderRef.current = new MediaRecorder(stream, {
                mimeType: "video/webm",
                videoBitsPerSecond: 2500000,
                audioBitsPerSecond: 128000,
            });

            mediaRecorderRef.current.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    recordedChunks.current.push(event.data);
                }
            };

            mediaRecorderRef.current.onstart = () => {
                setTimeout(() => {
                    setIsRecordingReady(true);
                }, 1000);
            };

            mediaRecorderRef.current.start();
            setIsRecording(true);
            startTimer();

        } catch (error) {
            setErrorMessage(`Произошла ошибка при начале записи: ${error.message}`);
            console.error("Ошибка при начале записи:", error);
        }
    };

    const resetMediaRecorder = () => {
        setIsRecording(false);
        if (videoRef.current && videoRef.current.srcObject) {
            const tracks = videoRef.current.srcObject.getTracks();
            tracks.forEach(track => track.stop());
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
        recordedChunks.current = [];
        setIsRecordingReady(false);
    };

    const stopRecording = async () => {
        if (!isRecording) return;

        try {
            if (!isRecordingReady) {
                setErrorMessage("Запись не была инициализирована должным образом. Пожалуйста, попробуйте ещё раз.");
                return;
            }

            clearInterval(timerRef.current);

            if (videoRef.current && videoRef.current.srcObject) {
                const stream = videoRef.current.srcObject;
                stream.getTracks().forEach(track => track.stop());
            }

            if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
                const recordingStopped = new Promise((resolve) => {
                    mediaRecorderRef.current.onstop = () => {
                        resolve();
                    };
                    mediaRecorderRef.current.stop();
                });
                await recordingStopped;
            }

            if (recordedChunks.current.length === 0) {
                setErrorMessage("Не удалось записать видео. Пожалуйста, попробуйте ещё раз.");
                return;
            }

            const blob = new Blob(recordedChunks.current, {type: 'video/webm'});
            await uploadVideo(blob);

        } catch (error) {
            setErrorMessage(`Произошла ошибка при остановке записи: ${error.message}`);
            console.error("Ошибка при остановке записи:", error);
        } finally {
            resetMediaRecorder();
        }
    };

    const uploadVideo = async (blob) => {
        try {
            const formData = new FormData();
            formData.append('video', blob, 'recorded-video.webm');
            formData.append('attemptId', attemptId);

            // Отправляем запрос и завершаем интервью
            const response = await fetchWithAuth(
                "/hr-interview/upload-video",
                {method: "POST", body: formData},
                token
            );

            if (response.ok) {
                setIsInterviewFinished(true);
            } else {
                const errorData = await response.json();
                setErrorMessage(`Ошибка при загрузке видео: ${errorData.error || 'Неизвестная ошибка'}`);
            }
        } catch (error) {
            setErrorMessage(`Произошла ошибка при загрузке видео: ${error.message}`);
            console.error("Ошибка при загрузке видео:", error);
        }
    };

    const finishInterview = () => {
        navigate(`/hr-interview/results/${attemptId}`);
    };

    const viewHrInterviewHistory = () => {
        navigate("/hr-interview/results");
    };

    // Стартовая страница
    if (!isInterviewStarted) {
        return (
            <div className="start-page">
                <h1>Симулятор HR-скрининга</h1>
                <p>
                    Этот тренажер поможет вам подготовиться к собеседованию.
                    Вам будет предложено ответить на один вопрос с записью видео.
                    На ответ у вас будет 1,5 минуты.
                </p>
                <button className="start-button" onClick={startInterview}>
                    Начать HR-собеседование
                </button>
                {errorMessage && <p className="error-message">{errorMessage}</p>}
                <button className="history-button" onClick={viewHrInterviewHistory}>
                    Посмотреть историю попыток
                </button>
            </div>
        );
    }

    // Страница с результатами интервью
    if (isInterviewFinished) {
        return (
            <div className="results-page">
                <h1>Собеседование завершено!</h1>
                <p>Ваш ответ был успешно записан и отправлен на обработку.</p>
                <button className="finish-button" onClick={finishInterview}>
                    Перейти к результатам
                </button>
            </div>
        );
    }

    // Страница с одним вопросом
    return (
        <div className="interview-container">
            <h1>HR-собеседование</h1>
            <div className="interview-layout">
                <div className="video-section">
                    <div className="video-container">
                        <video ref={videoRef} className="video-preview" muted/>
                        {isRecording && (
                            <div className="timer">
                                {Math.floor(timeLeft / 60)}:{(timeLeft % 60).toString().padStart(2, '0')}
                            </div>
                        )}
                    </div>

                    <div className="controls">
                        {!isRecording ? (
                            <button className="record-button" onClick={startRecording}>
                                Начать запись
                            </button>
                        ) : (
                            <button className="answer-button" onClick={stopRecording}>
                                Завершить ответ
                            </button>
                        )}
                    </div>
                </div>

                <div className="question-section">
                    <div className="question-card">
                        <h3>Вопрос</h3>
                        <p className="question-text">{question.text}</p>
                    </div>
                </div>
            </div>

            {errorMessage && <p className="error-message">{errorMessage}</p>}
        </div>
    );
};

export default InterviewSimulator;