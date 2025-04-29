"""Un cas d'exmple pour valider le pipeline."""

from uuid import uuid4

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from arkey import arkey


async def parkey(x: str) -> None:
    """Effectue un tour dans le graphe de digita et en imprime la réponse."""
    async with AsyncSqliteSaver.from_conn_string("./arkeystate.db") as db:
        config = {"configurable": {"thread_id": str(uuid4()) }}
        rsp = await arkey(db).ainvoke({"original": x}, config)
        print(rsp["response"])  # noqa: T201

async def sarkey(x: str) -> None:
    """Effectue un tour dans le graphe de digita et en imprime la réponse."""
    async with AsyncSqliteSaver.from_conn_string("./arkeystate.db") as db:
        config = {"configurable": {"thread_id": str(uuid4()) }}
        async for c in arkey(db).astream({"original": x}, config):
            print(c)  # noqa: T201


async def main() -> None:
    """Run the main entry point of the program."""
    await sarkey("""S'agit-il de Marneffe Joseph né en 1829 à Namur?
    A-t-il été emprisonné et si oui combien de temps? Merci pour votre réponse""")
    print("==" * 80)  # noqa: T201
    #await parkey("""
    #Bonjour et Meilleurs Voeux
    #Je recherche dans vos archives numérisés de Bruxelles, l'acte de mariage 2117 du 30/11/1901
    #entre WILMART Jean Eugène & PIRARD Anna Alexandrine.
    #Je ne trouve pas dernier trimestre 1901.
    #Bien à vous
    #""")
    #print("==" * 120)  # noqa: T201
    #parkey("""
    #Bonjour chères collègues,
    #Un lecteur nous a fait remarquer qu'une erreur s'était glissée dans la dénomination de la paroisse de Hérinnes 
    #sur notre site web (cf. capture). Après vérification, il s'agit bien de la paroisse Sainte-Aldegonde et non de la 
    #paroisse Saint-Martin.
    #
    #Pourriez-vous corriger cette erreur lors de la prochaine mise à jour?
    #Merci
    #""")
    #print("==" * 120)  # noqa: T201
    #await parkey("""
    #JE vous remerci mesieux je suis ravie de vous conetresurprise pour moi que jentent que vous etes lire
    #mon sutuation malque mon pere etait orphelin de -un anciens combattants et victimes de uerremoi 
    #je suis son filsque il a un pension en 1943 OU LE RECUE EN 1948 JARIVE PAS DE COMPRENDRE LARCHIVES 
    #DE FRANCEVOILA IL ETAIT ORPHELINT ET OUBLILLER ET SON PENSION ETAIT SE PRENDRE PAR LA PROCURATION DE 
    #SON GRAND PEREJE VOUS DIRE LORSQUE SON PERE ETAIT MORT AU CHAMP DUNE BATAILLE A TUNISIE EN 1943 PAR 
    #LENEMIE DE NAZIEHITLERELA FRANCE DE LARMEES FRANCAISE CE DONNEZ SON DROITS DE PENSIONVOILA QUE MON 
    #PERE ETAIT NEE EN 1943 ET NE CONNEZ RIEN MESIEUX ELLE NA RIEN CONNEZ LHISTOIRE DE SON PEREMESKINE-AMMAR-NEE 
    #EN 1910 A SETIF DEPARTEMENT DE CONSTANTINEALGERIE
    #""")
    #print("==" * 120)  # noqa: T201
    #parkey("""
    #Beste,
    #Ik heb nooit de mail ontvangen met activatielink, waardoor ik niet kan aanmelden.
    #
    #Alvast bedankt!
    #
    #Met vriendelijke groeten
    #
    #Ilse Seynaeve
    #""")
    #print("==" * 120)  # noqa: T201
    #parkey("""
    #Bonjour, je n'arrive plus à faire de recherches sur le site des AGR. J'obtiens le message 
    #"Ce compte n'a pas encore été activé. Veuillez cliquer sur le lien d'activatione envoyé par mail." 
    #mais je n'ai pas reçu ce mail! Ce blocage est récent (depuis le passage à AGATHA). 
    #On a changé le mot de passe mais le blocage persiste.
    #Merci de faire le nécessaire.
    #Bien à vous,
    #Anne Hainaux
    #""")
    #print("==" * 120)  # noqa: T201
    #parkey("""
    #Dans les archives des actes de décès de Menen il y a un manque de registres.
    #Pouvez-vous me dire si ce manque est définitif pour la période des décès de 1854 à 1897 ou si c'est parce que tous les actes ne sont pas encore numérisés
    #Je recherche 3 actes de décès
    #
    #1° LAMOUR Joanna Sophia décédée à Menen le 26 décembre 1865, fille de Joannes Philippus et de Eugenia Ludovica SOCAER, veuve de Joannes Bernardus POTTILIUS
    #2° LEROY Theresia Julia décédée à Menen le 26 janvier 1864, fille de Joannes Henricus et de Isabella Rosa VUYLSTEKE, veuve de Ignatius Josephs POTTILIUS
    #3° DEKERTE Joanna Theresia décédée à Menen le 27 novembre 1858, fille de Joannes et de Maria Josepha MULDER, veuve de Petrus Jacobus POTTILIUS
    #
    #MERCI de votre aide
    #Bien cordialement
    #
    #Brigitte d'halluin
    #Chemin du Chaneboux
    #Les Palennes
    #63410 LOUBEYRAT
    #""")
    #print("="*80)
    #parkey("Pour quelles raison l'on me refuse toutes les données que j'inscris l'URL vient du message envoyé par vos soins, je ne comprends pas")
    #print("="*80)
    #parkey("Pouvez-vous m'adresser l'acte de décès de ma grande tante PERROTEY Emma Germaine décédée le 23/3/1919 à VINKEM , WEST-VLAANDEREN mariée à TAMIGNIAUX Gérard Nicolas Joseph Ghislain le 5/7/1917 à PARIS 3ème. Je vous remercie.<br>Daniel DEUTSCH.<br>17 rue Edouard NIEUPORT 69008 LYON")

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
