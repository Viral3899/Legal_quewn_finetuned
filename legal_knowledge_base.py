#!/usr/bin/env python3
"""Legal knowledge base with correct IPC information."""

class LegalKnowledgeBase:
    """Simple knowledge base for common legal questions."""
    
    def __init__(self):
        self.knowledge = {
            "302": {
                "section": "IPC Section 302",
                "title": "Murder",
                "description": "Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
                "simple_explanation": "Section 302 of the Indian Penal Code deals with murder. Anyone who commits murder can be punished with the death penalty or life imprisonment, and may also be fined."
            },
            "304": {
                "section": "IPC Section 304",
                "title": "Culpable Homicide not amounting to Murder",
                "description": "Whoever commits culpable homicide not amounting to murder shall be punished with imprisonment for life, or with imprisonment of either description for a term which may extend to ten years, and shall also be liable to fine.",
                "simple_explanation": "Section 304 deals with culpable homicide that doesn't amount to murder. The punishment is life imprisonment or up to 10 years imprisonment, plus fine."
            },
            "307": {
                "section": "IPC Section 307",
                "title": "Attempt to Murder",
                "description": "Whoever does any act with such intention or knowledge, and under such circumstances that, if he by that act caused death, he would be guilty of murder, shall be punished with imprisonment of either description for a term which may extend to ten years, and shall also be liable to fine.",
                "simple_explanation": "Section 307 deals with attempt to murder. If someone tries to kill another person, they can be punished with up to 10 years imprisonment and fine."
            },
            "375": {
                "section": "IPC Section 375",
                "title": "Rape",
                "description": "A man is said to commit 'rape' if he—(a) penetrates his penis, to any extent, into the vagina, mouth, urethra or anus of a woman or makes her to do so with him or any other person; or (b) inserts, to any extent, any object or a part of the body, not being the penis, into the vagina, the urethra or anus of a woman or makes her to do so with him or any other person; or (c) manipulates any part of the body of a woman so as to cause penetration into the vagina, urethra or anus of a woman or makes her to do so with him or any other person; or (d) applies his mouth to the vagina, anus, urethra of a woman or makes her to do so with him or any other person, under the circumstances falling under any of the following seven descriptions:—(i) against her will; (ii) without her consent; (iii) with her consent, when her consent has been obtained by putting her or any person in whom she is interested in fear of death or of hurt; (iv) with her consent, when the man knows that he is not her husband, and that her consent is given because she believes that he is another man to whom she is or believes herself to be lawfully married; (v) with her consent, when, at the time of giving such consent, by reason of unsoundness of mind or intoxication or the administration by him personally or through another of any stupefying or unwholesome substance, she is unable to understand the nature and consequences of that to which she gives consent; (vi) with or without her consent, when she is under eighteen years of age; (vii) when she is unable to communicate consent.",
                "simple_explanation": "Section 375 of the Indian Penal Code defines rape. It includes various forms of sexual assault and penetration without consent. The punishment for rape is imprisonment for not less than seven years, which may extend to life imprisonment, and fine."
            },
            "420": {
                "section": "IPC Section 420",
                "title": "Cheating and Dishonestly Inducing Delivery of Property",
                "description": "Whoever cheats and thereby dishonestly induces the person deceived to deliver any property to any person, or to make, alter or destroy the whole or any part of a valuable security, or anything which is signed or sealed, and which is capable of being converted into a valuable security, shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine.",
                "simple_explanation": "Section 420 deals with cheating. If someone cheats another person and gets them to give up property or valuable documents, they can be punished with up to 7 years imprisonment and fine."
            }
        }
    
    def search(self, query: str) -> dict:
        """Search for legal information based on query."""
        query_lower = query.lower()
        
        # Look for section numbers
        for section_num, info in self.knowledge.items():
            if section_num in query_lower or f"section {section_num}" in query_lower:
                return {
                    "found": True,
                    "section": info["section"],
                    "title": info["title"],
                    "description": info["description"],
                    "simple_explanation": info["simple_explanation"]
                }
        
        # Look for keywords
        if "murder" in query_lower:
            return self.knowledge["302"]
        elif "culpable homicide" in query_lower:
            return self.knowledge["304"]
        elif "attempt to murder" in query_lower or "attempt murder" in query_lower:
            return self.knowledge["307"]
        elif "cheating" in query_lower:
            return self.knowledge["420"]
        
        return {"found": False, "message": "No matching legal information found."}
    
    def format_answer(self, query: str) -> str:
        """Format a proper answer for the query."""
        result = self.search(query)
        
        if not result["found"]:
            return "I apologize, but I don't have specific information about that legal topic. Please try asking about a specific IPC section number or legal term."
        
        answer = f"**{result['section']} - {result['title']}**\n\n"
        answer += f"**Legal Definition:** {result['description']}\n\n"
        answer += f"**Simple Explanation:** {result['simple_explanation']}"
        
        return answer

# Global instance
legal_kb = LegalKnowledgeBase()
