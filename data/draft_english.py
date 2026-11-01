"""Unreviewed English answer drafts for I01–I08. Do not train on these yet."""

ENGLISH_DRAFTS = {
    "I01": {
        "meaning": "'생각해 볼게요' means 'I'll think about it.'",
        "interpretation": "Your friend may need to check their plans or may want more time to decide.",
        "response": "Give them time. If you need an answer for a reservation, tell them when you need it.",
        "uncertainty": "This sentence alone is neither a definite yes nor a definite no.",
    },
    "I02": {
        "meaning": "'다음에 같이 가요' means 'Let's go together another time.'",
        "interpretation": "It may be a sincere suggestion for later, or a polite way to end the conversation.",
        "response": "If you would like to meet, suggest a specific date later.",
        "uncertainty": "Without a date, no plan has been confirmed.",
    },
    "I03": {
        "meaning": "In this situation, '괜찮아요' most likely means 'I'm okay, thank you.'",
        "interpretation": "The person is politely declining the seat you offered.",
        "response": "Respect their answer and do not keep insisting.",
        "uncertainty": "You cannot tell why they declined from this sentence alone.",
    },
    "I04": {
        "meaning": "'조금 어려울 것 같아요' means that the proposed time may not work for them.",
        "interpretation": "They may be politely declining that time rather than the meeting itself.",
        "response": "Ask whether another time would work.",
        "uncertainty": "The reason it is difficult is not clear from this sentence.",
    },
    "I05": {
        "meaning": "'언제 한번 봐요' means 'Let's meet sometime.'",
        "interpretation": "It may express warmth without setting an actual appointment.",
        "response": "If you want to make a plan, suggest a date and place.",
        "uncertainty": "No meeting time has been agreed yet.",
    },
    "I06": {
        "meaning": "'부담 갖지 마세요' means 'Please don't feel obligated.'",
        "interpretation": "The giver may be trying to make you feel comfortable accepting the gift.",
        "response": "Thank them and respond in a way that is comfortable for you.",
        "uncertainty": "You cannot know all of the giver's expectations from this phrase alone.",
    },
    "I07": {
        "meaning": "'시간 되면 들러요' means 'Drop by if you have time.'",
        "interpretation": "It is an invitation, not an obligation to attend.",
        "response": "If you are interested, check the event time and location.",
        "uncertainty": "The phrase does not tell you whether a reservation is needed.",
    },
    "I08": {
        "meaning": "'나중에 연락드릴게요' means 'I'll contact you later.'",
        "interpretation": "The staff member has not given you a result or a precise contact date.",
        "response": "Follow the stated timeline if there is one; otherwise, ask when to expect an update.",
        "uncertainty": "It does not imply that your application was accepted.",
    },
}
